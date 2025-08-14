# @plawanrath

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

from transformers import DetrForObjectDetection

# Your repo imports
from src.datasets import CocoDetectionDataset
from src.models import MaskedAutoencoderViT
from src.utils import get_coco_api_from_dataset

# --------------------- MAE backbone adapter --------------------- #

class MAEBackbone(nn.Module):
    """
    Wrap a ViT/MAE encoder and emit a single feature map as a list [Tensor(B,C,H',W')].
    Conforms to HF DETR backbone signature: forward(pixel_values, pixel_mask) -> (features_list, object_queries_list)
    """
    def __init__(self, mae_encoder: nn.Module, embed_dim: int, out_channels: int, patch_size: int = 16):
        super().__init__()
        self.encoder = mae_encoder
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        # HF DETR reads this to size its input projection conv weights
        self.num_channels = [out_channels]

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "forward_features"):
            tokens = self.encoder.forward_features(x)
        else:
            tokens = self.encoder(x)

        if isinstance(tokens, dict):
            if "x" in tokens:
                tokens = tokens["x"]
            elif "last_hidden_state" in tokens:
                tokens = tokens["last_hidden_state"]
            else:
                for v in tokens.values():
                    if torch.is_tensor(v):
                        tokens = v
                        break

        assert torch.is_tensor(tokens), "Backbone encoder didn't return a tensor or known dict field."

        # Drop CLS if N+1 looks like square + 1
        if tokens.dim() == 3 and tokens.size(1) > 0:
            n = tokens.size(1)
            root = int((n - 1) ** 0.5)
            if root * root + 1 == n:
                tokens = tokens[:, 1:, :]
        return tokens  # (B, N, D)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor | None = None):
        B, _, H, W = pixel_values.shape
        h = H // self.patch_size
        w = W // self.patch_size

        tokens = self._encode_tokens(pixel_values)  # (B, N, D)
        B2, N, D = tokens.shape
        assert B2 == B, "Batch mismatch between input and encoder output."
        if N != h * w:
            root = int(N ** 0.5)
            if root * root == N:
                h = w = root
            else:
                raise ValueError(f"Cannot reshape tokens of length {N} to (H={h}, W={w}).")

        feat = tokens.transpose(1, 2).contiguous().view(B, D, h, w)  # (B, D, H', W')
        feat = self.proj(feat)                             # (B, out_channels, H', W')
        
        # Create mask for the feature map
        if pixel_mask is not None:
            # Downsample the pixel mask to match feature dimensions
            mask = torch.nn.functional.interpolate(
                pixel_mask.unsqueeze(1).float(), 
                size=(h, w), 
                mode='nearest'
            ).squeeze(1).bool()
        else:
            # Create a mask of all True (no masking)
            mask = torch.ones((B, h, w), dtype=torch.bool, device=feat.device)
        
        # Create positional embeddings with d_model channels (256), not backbone channels (2048)
        # This matches what the original DETR backbone returns
        d_model = 256  # DETR's model dimension
        
        # Create simple 2D positional embeddings (sine/cosine embeddings like in DETR)
        pos_embed = torch.zeros((B, d_model, h, w), dtype=feat.dtype, device=feat.device)
        
        # Simple positional encoding
        y_pos = torch.arange(h, dtype=torch.float32, device=feat.device).unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(w, dtype=torch.float32, device=feat.device).unsqueeze(0).repeat(h, 1)
        
        # Normalize to [0, 2*pi)
        y_pos = y_pos / h * 6.28318
        x_pos = x_pos / w * 6.28318
        
        # Create frequency components
        dim_t = torch.arange(d_model // 4, dtype=torch.float32, device=feat.device)
        dim_t = 10000 ** (2 * dim_t / (d_model // 4))
        
        pos_x = x_pos.unsqueeze(0).unsqueeze(0) / dim_t.view(1, -1, 1, 1)
        pos_y = y_pos.unsqueeze(0).unsqueeze(0) / dim_t.view(1, -1, 1, 1)
        
        # Fill positional embeddings
        pos_embed[:, 0::4, :, :] = torch.sin(pos_x).expand(B, -1, -1, -1)
        pos_embed[:, 1::4, :, :] = torch.cos(pos_x).expand(B, -1, -1, -1)
        pos_embed[:, 2::4, :, :] = torch.sin(pos_y).expand(B, -1, -1, -1)
        pos_embed[:, 3::4, :, :] = torch.cos(pos_y).expand(B, -1, -1, -1)
        
        return [(feat, mask)], [pos_embed]

# --------------------- helpers --------------------- #

def xyxy_to_cxcywh_norm(boxes_xyxy: torch.Tensor, in_w: int, in_h: int, out_w: int, out_h: int) -> torch.Tensor:
    """
    Convert boxes from xyxy (absolute in original image space) to
    cxcywh normalized in [0,1] relative to the resized image (out_w,out_h).
    """
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.new_zeros((0, 4))

    sx = out_w / float(in_w)
    sy = out_h / float(in_h)
    boxes = boxes_xyxy.clone()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy

    x_min, y_min, x_max, y_max = boxes.unbind(dim=1)
    w = (x_max - x_min).clamp(min=1e-6)
    h = (y_max - y_min).clamp(min=1e-6)
    cx = x_min + 0.5 * w
    cy = y_min + 0.5 * h

    return torch.stack([cx / out_w, cy / out_h, w / out_w, h / out_h], dim=1)

def build_label_mappings(coco_api):
    cat_ids = coco_api.getCatIds()
    cats = coco_api.loadCats(cat_ids)
    cats_sorted = sorted(cats, key=lambda x: x["id"])
    id2label = {i: c["name"] for i, c in enumerate(cats_sorted)}
    label2id = {v: k for k, v in id2label.items()}
    cat_id_to_contig = {c["id"]: i for i, c in enumerate(cats_sorted)}  # raw COCO id -> contiguous idx
    return id2label, label2id, cat_id_to_contig, len(cats_sorted)

def collate_keep(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def infer_backbone_in_channels(detr_model: nn.Module, d_model: int) -> int:
    """
    Find the input projection layer that projects backbone features -> d_model.
    Returns its in_channels. Fallback to 2048 for ResNet-50 DETR.
    """
    # Look for the input_projection layer in the model
    for name, m in detr_model.named_modules():
        if 'input_projection' in name and isinstance(m, nn.Conv2d):
            if hasattr(m, 'in_channels') and hasattr(m, 'out_channels'):
                if getattr(m, "out_channels", None) == d_model:
                    return int(m.in_channels)
    
    # Final fallback for facebook/detr-resnet-50 family
    return 2048

# --------------------- main --------------------- #

def main(args):
    # ---- setup ----
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- dataset ----
    # IMPORTANT: your CocoDetectionDataset must return:
    #   target["boxes"] as FloatTensor[N,4] in XYXY (original image scale)
    #   target["labels"] as LongTensor[N] of raw COCO category_id values
    train_dataset = CocoDetectionDataset(
        root=Path(args.data_path) / "train/images",
        ann_file=Path(args.data_path) / "train/annotations.json",
        transform=None  # we resize/normalize later and scale boxes accordingly
    )

    coco_api = get_coco_api_from_dataset(train_dataset)
    id2label, label2id, cat_id_to_contig, num_classes = build_label_mappings(coco_api)
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

    # image preproc to 224x224 (match MAE ViT-S/16)
    img_preproc = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    out_h, out_w = 224, 224

    # ---- model ----
    model = DetrForObjectDetection.from_pretrained(
        args.base_detr_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id,
    )

    # Determine what channel size DETR's input_proj expects (e.g., 2048 for ResNet-50)
    d_model = model.config.d_model
    expected_in = infer_backbone_in_channels(model, d_model)
    print(f"DETR d_model: {d_model}, expected backbone channels: {expected_in}")

    mae_full = MaskedAutoencoderViT(model_name='vit_small_patch16_224')
    mae_encoder = mae_full.encoder

    print(f"Loading MAE encoder weights from {args.pretrained_encoder}")
    state = torch.load(args.pretrained_encoder, map_location="cpu")
    missing, unexpected = mae_encoder.load_state_dict(state, strict=False)
    print(f"[MAE load] missing keys: {len(missing)} | unexpected: {len(unexpected)}")

    embed_dim = getattr(mae_encoder, "embed_dim", 384)  # ViT-S/16 typical
    patch_size = 16
    print(f"MAE encoder embed_dim: {embed_dim}, patch_size: {patch_size}")
    backbone = MAEBackbone(
        mae_encoder,
        embed_dim=embed_dim,
        out_channels=expected_in,   # IMPORTANT: match DETR's expected backbone channels
        patch_size=patch_size
    )
    print(f"MAE backbone output channels: {backbone.out_channels}")
    model.model.backbone = backbone
    model.to(device)

    # ---- optim ----
    def _is_backbone(n): return "model.backbone" in n or n.startswith("model.backbone")
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if not _is_backbone(n) and p.requires_grad], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if _is_backbone(n) and p.requires_grad], "lr": args.lr_backbone},
    ]
    optimizer = optim.AdamW(param_dicts, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # ---- train ----
    print("Starting fine-tuning (manual batching, no HF processor)...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for step, (images, targets) in enumerate(train_loader, start=1):
            batch_pixels = []
            batch_labels = []

            for img, tgt in zip(images, targets):
                # image dims
                if hasattr(img, "size"):
                    in_w, in_h = img.size  # PIL: (W, H)
                else:
                    _, in_h, in_w = img.shape

                # preprocess image
                img_t = img_preproc(img)  # (3,224,224)

                # boxes: xyxy (original scale) -> cxcywh normalized to 224x224
                boxes_xyxy = tgt["boxes"]  # FloatTensor [N,4]
                boxes_cxcywh = xyxy_to_cxcywh_norm(
                    boxes_xyxy, in_w=in_w, in_h=in_h, out_w=out_w, out_h=out_h
                )

                # labels: raw COCO category_id -> contiguous 0..K-1
                raw_labels = tgt["labels"].tolist()
                class_labels = torch.as_tensor(
                    [cat_id_to_contig[int(cid)] for cid in raw_labels],
                    dtype=torch.int64
                )

                batch_pixels.append(img_t)
                batch_labels.append({
                    "class_labels": class_labels,
                    "boxes": boxes_cxcywh.to(dtype=torch.float32),
                })

            pixel_values = torch.stack(batch_pixels, dim=0).to(device)  # (B,3,224,224)
            pixel_mask = torch.ones(
                (pixel_values.size(0), out_h, out_w),
                dtype=torch.bool,
                device=device
            )  # no padding since all 224x224
            labels = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}
                      for d in batch_labels]

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

        # Save checkpoint
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

# --------------------- CLI --------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DETR Fine-tuning with MAE Backbone (XYXY dataset, no HF processor)')
    parser.add_argument('--data_path', type=str, default='data/finetune',
                        help='Path to COCO-format data dir (expects train/images + train/annotations.json)')
    parser.add_argument('--pretrained_encoder', type=str, required=True,
                        help='Path to MAE encoder .pth (e.g., checkpoints/mae_pretrained/mae_vit_encoder_epoch_180.pth)')
    parser.add_argument('--base_detr_id', type=str, default='facebook/detr-resnet-50',
                        help='HF model id or local dir for the DETR base')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/detr_finetuned',
                        help='Where to save fine-tuned checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='(reserved) Where to save visualizations/plots')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (DETR is memory heavy)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (transformer/head)')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help='Learning rate (MAE backbone)')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Epoch interval to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='Steps between loss logs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers (will be forced to 0 on MPS)')

    args = parser.parse_args()
    main(args)
