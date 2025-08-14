# @plawanrath

import torch
import torch.nn as nn
import timm
from transformers import DetrConfig, DetrForObjectDetection


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with VisionTransformer backbone.
    """
    def __init__(self, model_name='vit_small_patch16_224', mask_ratio=0.75):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        
        # 1. ViT Encoder (Backbone)
        self.encoder = timm.create_model(model_name, pretrained=False)
        self.patch_size = self.encoder.patch_embed.patch_size[0] if isinstance(self.encoder.patch_embed.patch_size, tuple) else self.encoder.patch_embed.patch_size
        
        # 2. MAE Decoder
        encoder_dim = self.encoder.embed_dim
        decoder_dim = 512 # A common choice, smaller than encoder_dim
        num_patches = self.encoder.patch_embed.num_patches
        
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Positional embedding for the decoder
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False)
        
        # A simple transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=16, dim_feedforward=2048, activation='gelu', batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=8)
        
        # 3. Reconstruction Head
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_size**2 * 3, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialization similar to the original MAE paper
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # Initialize decoder positional embedding from encoder
        pos_embed = self.encoder.pos_embed
        decoder_pos_embed = self.decoder_pos_embed
        
        # Simple copy if shapes match, otherwise initialize with zeros
        if pos_embed.shape == decoder_pos_embed.shape:
            decoder_pos_embed.data.copy_(pos_embed.data)
        else:
            torch.nn.init.normal_(decoder_pos_embed, std=.02)

    def random_masking(self, x):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        
        # Sort noise and find indices to keep and restore
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        # Embed patches
        x = self.encoder.patch_embed(x)
        
        # Add position embedding
        x = x + self.encoder.pos_embed[:, 1:, :] # Skip CLS token pos embed
        
        # Masking
        x, mask, ids_restore = self.random_masking(x)
        
        # Append cls token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Transformer blocks
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        N = x.shape[0]
        mask_tokens = self.mask_token.repeat(N, ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # Add pos embed
        x = x + self.decoder_pos_embed
        
        # Apply Transformer blocks
        x = self.decoder(x)
        
        # Predictor
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs:
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  # Mean loss on removed patches
        return loss

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def create_detr_model_with_custom_backbone(num_classes, pretrained_encoder_path):
    """
    Creates a DETR model with a ViT-Small backbone and loads pre-trained encoder weights.
    
    Args:
        num_classes (int): The number of object classes for the detection head.
        pretrained_encoder_path (str): Path to the saved.pth file for the MAE encoder.
        
    Returns:
        A DetrForObjectDetection model with the custom, pre-trained backbone.
    """
    # 1. Define the configuration
    config = DetrConfig(
        num_labels=num_classes,
        use_timm_backbone=True,
        backbone="vit_small_patch16_224",
        backbone_config=None, # Not needed when use_timm_backbone is True
        num_queries=100, # Standard number of object queries for DETR
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
    )

    # 2. Instantiate the model with the custom config
    # ignore_mismatched_sizes=True is crucial as we are replacing the classification head
    model = DetrForObjectDetection.from_pretrained(
        None, 
        config=config, 
        ignore_mismatched_sizes=True
    )

    # 3. Load the pre-trained encoder weights
    if pretrained_encoder_path:
        print(f"Loading pre-trained encoder weights from: {pretrained_encoder_path}")
        encoder_state_dict = torch.load(pretrained_encoder_path, map_location='cpu')
        
        # The backbone is nested within model.model.backbone
        model.model.backbone.load_state_dict(encoder_state_dict)
        print("Successfully loaded pre-trained backbone weights.")
    else:
        print("Warning: No pre-trained encoder path provided. Initializing backbone with random weights.")

    return model