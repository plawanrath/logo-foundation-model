# @plawanrath
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from torchvision import transforms as T
from transformers import DetrForObjectDetection

# ---------- MAE backbone (same as train) ----------
class MAEBackbone(nn.Module):
    def __init__(self, mae_encoder: nn.Module, embed_dim: int, out_channels: int, d_model: int = 256, patch_size: int = 16):
        super().__init__()
        self.encoder = mae_encoder
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.d_model = d_model
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.num_channels = [out_channels]  # read by HF DETR

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "forward_features"):
            tokens = self.encoder.forward_features(x)
        else:
            tokens = self.encoder(x)
        if isinstance(tokens, dict):
            if "x" in tokens: tokens = tokens["x"]
            elif "last_hidden_state" in tokens: tokens = tokens["last_hidden_state"]
            else:
                for v in tokens.values():
                    if torch.is_tensor(v):
                        tokens = v; break
        if not torch.is_tensor(tokens):
            raise RuntimeError("MAEBackbone: encoder didn't return a tensor.")
        # drop CLS if present
        if tokens.dim() == 3 and tokens.size(1) > 0:
            n = tokens.size(1); r = int((n - 1) ** 0.5)
            if r * r + 1 == n: tokens = tokens[:, 1:, :]
        return tokens  # (B, N, D)

    def _positional_encoding(self, B: int, h: int, w: int, device, dtype):
        d_model = self.d_model
        pos = torch.zeros((B, d_model, h, w), dtype=dtype, device=device)
        y = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, w)
        x = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(0).repeat(h, 1)
        y = y / max(1, h) * 6.28318; x = x / max(1, w) * 6.28318  # 2π
        dim_t = torch.arange(d_model // 4, dtype=torch.float32, device=device)
        dim_t = 10000 ** (2 * dim_t / (d_model // 4))
        pos_x = x.unsqueeze(0).unsqueeze(0) / dim_t.view(1, -1, 1, 1)
        pos_y = y.unsqueeze(0).unsqueeze(0) / dim_t.view(1, -1, 1, 1)
        pos[:, 0::4] = torch.sin(pos_x).expand(B, -1, -1, -1)
        pos[:, 1::4] = torch.cos(pos_x).expand(B, -1, -1, -1)
        pos[:, 2::4] = torch.sin(pos_y).expand(B, -1, -1, -1)
        pos[:, 3::4] = torch.cos(pos_y).expand(B, -1, -1, -1)
        return pos

    def forward(self, pixel_values: torch.Tensor, pixel_mask: Optional[torch.Tensor] = None):
        B, _, H, W = pixel_values.shape
        h, w = H // self.patch_size, W // self.patch_size
        tokens = self._encode_tokens(pixel_values)
        B2, N, D = tokens.shape
        if B2 != B: raise RuntimeError("Batch mismatch.")
        if N != h * w:
            r = int(N ** 0.5)
            if r * r == N: h = w = r
            else: raise ValueError(f"Cannot reshape N={N} to {h}x{w}.")
        feat = tokens.transpose(1, 2).contiguous().view(B, D, h, w)
        feat = self.proj(feat)  # (B, out_channels, h, w)
        if pixel_mask is not None:
            mask = torch.nn.functional.interpolate(pixel_mask.unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze(1).bool()
        else:
            mask = torch.ones((B, h, w), dtype=torch.bool, device=feat.device)
        pos = self._positional_encoding(B, h, w, feat.device, feat.dtype)
        return [(feat, mask)], [pos]


# ---------- Shared helpers ----------
def get_image_preproc(size_hw: Tuple[int, int]=(224,224)) -> T.Compose:
    return T.Compose([
        T.Resize(size_hw),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def collate_keep(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

def build_label_mappings(coco_api):
    cat_ids = coco_api.getCatIds()
    cats = coco_api.loadCats(cat_ids)
    cats_sorted = sorted(cats, key=lambda x: x["id"])
    id2label = {i: c["name"] for i, c in enumerate(cats_sorted)}
    label2id = {v: k for k, v in id2label.items()}
    cat_id_to_contig = {c["id"]: i for i, c in enumerate(cats_sorted)}
    contig_to_cat_id = {v: k for k, v in cat_id_to_contig.items()}
    return id2label, label2id, cat_id_to_contig, contig_to_cat_id, len(cats_sorted)

def xyxy_to_cxcywh_norm(boxes_xyxy: torch.Tensor, in_w: int, in_h: int, out_w: int, out_h: int) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.new_zeros((0, 4))
    sx, sy = out_w / float(in_w), out_h / float(in_h)
    boxes = boxes_xyxy.clone()
    boxes[:, [0, 2]] *= sx; boxes[:, [1, 3]] *= sy
    x0, y0, x1, y1 = boxes.unbind(1)
    w = (x1 - x0).clamp(min=1e-6); h = (y1 - y0).clamp(min=1e-6)
    cx, cy = x0 + 0.5 * w, y0 + 0.5 * h
    return torch.stack([cx/out_w, cy/out_h, w/out_w, h/out_h], 1)

def postprocess_detr(outputs, target_sizes: List[Tuple[int,int]], score_thresh: float):
    logits = outputs.logits
    prob = logits.softmax(-1)
    scores, labels = prob[..., :-1].max(-1)  # drop "no-object"
    boxes = outputs.pred_boxes  # normalized cxcywh
    res = []
    for b, (H,W) in enumerate(target_sizes):
        scale = torch.tensor([W,H,W,H], device=boxes.device, dtype=boxes.dtype)
        bxs = boxes[b] * scale
        xywh = bxs.clone()
        xywh[:,0] = bxs[:,0] - 0.5*bxs[:,2]
        xywh[:,1] = bxs[:,1] - 0.5*bxs[:,3]
        keep = scores[b] > score_thresh
        res.append({"scores": scores[b][keep], "labels": labels[b][keep], "boxes": xywh[keep]})
    return res


# ---------- Checkpoint I/O ----------
def _load_file_any(fp: Path):
    if fp.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as safe_load
        except ImportError as e:
            raise RuntimeError("Install safetensors: pip install safetensors") from e
        return safe_load(str(fp))
    return torch.load(fp, map_location="cpu", weights_only=False)

def read_state_dict(model_path: str | Path) -> Dict[str, torch.Tensor]:
    p = Path(model_path)
    if p.is_file():
        return _load_file_any(p)
    if p.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin"):
            fp = p / name
            if fp.exists(): return _load_file_any(fp)
        # shards
        shard_pat = re.compile(r"(model|pytorch_model)-\d{5}-of-\d{5}\.(safetensors|bin)")
        shards = sorted([fp for fp in p.iterdir() if shard_pat.match(fp.name)])
        if shards:
            state = {}
            for s in shards:
                state.update(_load_file_any(s))
            return state
        raise FileNotFoundError(f"No weights in {p} (looked for model.safetensors / pytorch_model.bin / shards)")
    raise FileNotFoundError(f"Path does not exist: {p}")

def get_backbone_out_channels_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    w = state.get("model.backbone.proj.weight", None)
    if w is not None and w.ndim == 4:
        return int(w.shape[0])  # out_channels
    return None

def infer_backbone_in_channels_from_model(detr_model: nn.Module, d_model: int) -> int:
    # Scan 1x1 convs with out_channels==d_model; return their in_channels (fallback 2048).
    for m in detr_model.modules():
        if isinstance(m, nn.Conv2d):
            if getattr(m, "kernel_size", None) == (1,1) and getattr(m, "out_channels", None) == d_model:
                return int(m.in_channels)
    return 2048

def build_mae_detr(num_classes: int,
                   base_detr_id: str = "facebook/detr-resnet-50",
                   mae_encoder_weights_path: Optional[str] = None,
                   mae_model_name: str = "vit_small_patch16_224",
                   patch_size: int = 16,
                   override_backbone_out_channels: Optional[int] = None) -> DetrForObjectDetection:
    model = DetrForObjectDetection.from_pretrained(
        base_detr_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    from src.models import MaskedAutoencoderViT
    mae_full = MaskedAutoencoderViT(model_name=mae_model_name)
    mae_encoder = mae_full.encoder

    if mae_encoder_weights_path:
        enc_state = torch.load(mae_encoder_weights_path, map_location="cpu")
        mae_encoder.load_state_dict(enc_state, strict=False)

    d_model = model.config.d_model
    expected_in = override_backbone_out_channels
    if expected_in is None:
        expected_in = infer_backbone_in_channels_from_model(model, d_model)

    backbone = MAEBackbone(
        mae_encoder=mae_encoder,
        embed_dim=getattr(mae_encoder, "embed_dim", 384),
        out_channels=expected_in,
        d_model=d_model,
        patch_size=patch_size,
    )
    model.model.backbone = backbone
    return model

def load_finetuned_weights(model: nn.Module, model_path_or_state):
    if isinstance(model_path_or_state, (str, Path)):
        state = read_state_dict(model_path_or_state)
    elif isinstance(model_path_or_state, dict):
        state = model_path_or_state
    else:
        raise TypeError("load_finetuned_weights expects a path or a state-dict.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing: {len(missing)} | unexpected: {len(unexpected)}")
