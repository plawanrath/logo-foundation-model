# @plawanrath

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import CocoDetectionDataset           # must return xyxy boxes + raw category_id labels
from src.utils import get_coco_api_from_dataset
from src.train_eval_utils import (
    build_mae_detr,
    build_label_mappings,
    get_image_preproc,
    xyxy_to_cxcywh_norm,
    collate_keep,
)

def main(args):
    # ---- setup ----
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- dataset ----
    train_dataset = CocoDetectionDataset(
        root=Path(args.data_path) / "train/images",
        ann_file=Path(args.data_path) / "train/annotations.json",
        transform=None
    )
    coco_api = get_coco_api_from_dataset(train_dataset)
    id2label, label2id, cat_id_to_contig, _contig_to_cat_id, num_classes = (*build_label_mappings(coco_api),)[:5]
    print(f"Dataset has {num_classes} classes.")

    num_workers = 0 if device.type == "mps" else max(2, args.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_keep,
        pin_memory=(device.type in {"cuda", "mps"})
    )

    img_preproc = get_image_preproc((args.resize_h, args.resize_w))
    out_h, out_w = args.resize_h, args.resize_w

    # ---- model ----
    model = build_mae_detr(
        num_classes=num_classes,
        base_detr_id=args.base_detr_id,
        mae_encoder_weights_path=args.pretrained_encoder,  # load MAE here for finetuning
        mae_model_name='vit_small_patch16_224',
        patch_size=16,
    ).to(device)

    # ---- optim ----
    def _is_backbone(n): return "model.backbone" in n or n.startswith("model.backbone")
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if not _is_backbone(n) and p.requires_grad], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if _is_backbone(n) and p.requires_grad], "lr": args.lr_backbone},
    ]
    optimizer = optim.AdamW(param_dicts, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # ---- train ----
    print("Starting fine-tuning...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for step, (images, targets) in enumerate(train_loader, start=1):
            batch_pixels = []
            batch_labels = []
            for img, tgt in zip(images, targets):
                # image dims
                if hasattr(img, "size"):
                    in_w, in_h = img.size
                else:
                    _, in_h, in_w = img.shape

                img_t = img_preproc(img)
                boxes_xyxy = tgt["boxes"]
                boxes_cxcywh = xyxy_to_cxcywh_norm(boxes_xyxy, in_w, in_h, out_w, out_h)

                raw_labels = tgt["labels"].tolist()
                class_labels = torch.as_tensor([cat_id_to_contig[int(cid)] for cid in raw_labels], dtype=torch.int64)

                batch_pixels.append(img_t)
                batch_labels.append({"class_labels": class_labels, "boxes": boxes_cxcywh.to(torch.float32)})

            pixel_values = torch.stack(batch_pixels, dim=0).to(device)
            pixel_mask = torch.ones((pixel_values.size(0), out_h, out_w), dtype=torch.bool, device=device)
            labels = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()} for d in batch_labels]

            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach())
            if step % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / max(1, len(train_loader))
        print(f"--- End of Epoch [{epoch+1}/{args.epochs}] | Avg Loss: {avg_epoch_loss:.4f} ---")

        lr_scheduler.step()

        if (epoch + 1) % args.save_interval == 0:
            ckpt_dir = Path(args.checkpoint_dir) / f"epoch_{epoch+1:03d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir.as_posix())
            print(f"Saved HF checkpoint to {ckpt_dir}")

    print("Fine-tuning complete.")
    final_dir = Path(args.checkpoint_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir.as_posix())
    print(f"Saved final HF checkpoint to {final_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DETR Fine-tuning with MAE Backbone (XYXY dataset)')
    parser.add_argument('--data_path', type=str, default='data/finetune')
    parser.add_argument('--pretrained_encoder', type=str, required=True,
                        help='Path to MAE encoder .pth (e.g., checkpoints/mae_pretrained/mae_vit_encoder_epoch_180.pth)')
    parser.add_argument('--base_detr_id', type=str, default='facebook/detr-resnet-50')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/detr_finetuned')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resize_h', type=int, default=224)
    parser.add_argument('--resize_w', type=int, default=224)

    args = parser.parse_args()
    main(args)