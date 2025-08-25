# @plawanrath

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval

# Your repo imports
from src.datasets import CocoDetectionDataset           # returns xyxy boxes + raw category_id labels
from src.utils import get_coco_api_from_dataset

# Shared utils (ensure your src/eval_utils.py has the latest helpers we discussed)
from src.eval_utils import (
    build_mae_detr,
    build_label_mappings,
    get_image_preproc,
    postprocess_detr,
    collate_keep,
    read_state_dict,
    get_backbone_out_channels_from_state,
    load_finetuned_weights,
)

def run_inference(model, device, data_loader, img_preproc, out_h, out_w, contig_to_cat_id, score_thresh):
    """Run a full pass and return a flat COCO JSON list and total detections count."""
    results_json = []
    total_dets = 0
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc=f"Evaluating @th={score_thresh:.3f}"):
            batch_pixels = []
            tgt_sizes = []
            img_ids = []

            for img, tgt in zip(images, targets):
                # original (W, H)
                if hasattr(img, "size"):
                    in_w, in_h = img.size
                else:
                    _, in_h, in_w = img.shape

                # ids & sizes in (H, W) for post-process scaling
                img_ids.append(int(tgt["image_id"].item()) if torch.is_tensor(tgt["image_id"]) else int(tgt["image_id"]))
                tgt_sizes.append((in_h, in_w))

                # preprocess to match training
                batch_pixels.append(img_preproc(img))

            pixel_values = torch.stack(batch_pixels, dim=0).to(device)
            pixel_mask = torch.ones((pixel_values.size(0), out_h, out_w), dtype=torch.bool, device=device)

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            proc = postprocess_detr(outputs, target_sizes=tgt_sizes, score_thresh=score_thresh)

            # Convert to COCO json rows
            for b, out in enumerate(proc):
                img_id = img_ids[b]
                scores_b = out["scores"].tolist()
                labels_b = out["labels"].tolist()     # contiguous 0..K-1
                boxes_b = out["boxes"].tolist()       # xywh absolute

                total_dets += len(scores_b)
                for score, contig_label, xywh in zip(scores_b, labels_b, boxes_b):
                    cat_id = contig_to_cat_id[int(contig_label)]
                    x, y, w, h = xywh
                    results_json.append({
                        "image_id": img_id,
                        "category_id": int(cat_id),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(score),
                    })
    return results_json, total_dets

def main(args):
    # ---- setup ----
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- dataset (VAL) ----
    val_dataset = CocoDetectionDataset(
        root=Path(args.data_path) / "val/images",
        ann_file=Path(args.data_path) / "val/annotations.json",
        transform=None
    )

    coco_gt = get_coco_api_from_dataset(val_dataset)
    id2label, label2id, cat_id_to_contig, contig_to_cat_id, num_classes = build_label_mappings(coco_gt)
    print(f"Validation set classes: {num_classes}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if device.type == "mps" else args.num_workers,
        collate_fn=collate_keep,
        pin_memory=(device.type in {"cuda", "mps"}),
    )

    img_preproc = get_image_preproc((args.resize_h, args.resize_w))
    out_h, out_w = args.resize_h, args.resize_w

    # ---- model (build to match checkpoint → then load weights) ----
    # 1) Read checkpoint (dir or file); supports .safetensors/.bin/.pth/shards
    state = read_state_dict(args.model_path)
    # 2) Derive MAE backbone proj.out_channels from checkpoint to avoid shape mismatches
    out_ch = get_backbone_out_channels_from_state(state)
    print(f"[eval] checkpoint backbone proj.out_channels = {out_ch}")

    # 3) Build model (do NOT load MAE .pth here; use finetuned weights we just read)
    model = build_mae_detr(
        num_classes=num_classes,
        base_detr_id=args.base_detr_id,
        mae_encoder_weights_path=None,            # finetuned checkpoint already has weights
        mae_model_name='vit_small_patch16_224',
        patch_size=16,
        override_backbone_out_channels=out_ch,    # critical: match finetuned backbone channels
    ).to(device)

    # 4) Load finetuned weights
    load_finetuned_weights(model, state)
    model.eval()

    # ---- first pass at requested threshold ----
    results_json, total_dets = run_inference(
        model=model,
        device=device,
        data_loader=val_loader,
        img_preproc=img_preproc,
        out_h=out_h,
        out_w=out_w,
        contig_to_cat_id=contig_to_cat_id,
        score_thresh=args.score_thresh,
    )
    print(f"Detections @th={args.score_thresh:.3f}: {total_dets}")

    # ---- auto-lower threshold if empty (avoid COCOeval crash) ----
    if total_dets == 0 and args.auto_lower_thresh:
        fallback = args.fallback_thresh
        print(f"No detections found. Re-running once with a lower threshold: {fallback}")
        results_json, total_dets = run_inference(
            model=model,
            device=device,
            data_loader=val_loader,
            img_preproc=img_preproc,
            out_h=out_h,
            out_w=out_w,
            contig_to_cat_id=contig_to_cat_id,
            score_thresh=fallback,
        )
        print(f"Detections @th={fallback:.3f}: {total_dets}")

    # ---- if still empty, save empty JSON and exit gracefully ----
    results_path = Path(args.output_dir) / "predictions.json"
    if total_dets == 0:
        with open(results_path, "w") as f:
            json.dump([], f)
        print(f"Saved empty predictions to {results_path}")
        print("No detections produced. Try lowering --score_thresh further or check your training.")
        return

    # ---- save + run COCO eval ----
    with open(results_path, "w") as f:
        json.dump(results_json, f)
    print(f"Saved predictions to {results_path}")

    coco_dt = coco_gt.loadRes(str(results_path))  # safe now; non-empty
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # coco_eval.stats[0] is mAP@[.5:.95]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DETR Evaluation (MAE backbone, XYXY dataset)')
    parser.add_argument('--data_path', type=str, default='data/finetune',
                        help='Path to COCO-val dir (expects val/images + val/annotations.json)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to finetuned checkpoint dir or file (.safetensors/.bin/.pth or shards)')
    parser.add_argument('--base_detr_id', type=str, default='facebook/detr-resnet-50',
                        help='HF base model id used for finetune')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Where to save predictions.json')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--score_thresh', type=float, default=0.5)
    parser.add_argument('--auto_lower_thresh', action='store_true',
                        help='If set, auto-lower threshold once if no detections are found')
    parser.add_argument('--fallback_thresh', type=float, default=0.01,
                        help='Threshold to retry with if auto-lowering is enabled and results are empty')
    parser.add_argument('--resize_h', type=int, default=224)
    parser.add_argument('--resize_w', type=int, default=224)

    args = parser.parse_args()
    main(args)
