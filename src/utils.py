# @plawanrath

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image

def plot_loss_curve(losses, save_path):
    """
    Plots the training loss curve and saves it to a file.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('MAE Pre-training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")

def unpatchify(x, patch_size=16):
    """
    x: (N, L, patch_size**2 * 3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape**0.5)
    assert h * w == x.shape
    
    x = x.reshape(shape=(x.shape, h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape, 3, h * p, w * p))
    return imgs

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalizes a tensor image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_mae_reconstruction(imgs, pred, mask, save_path, num_images=8):
    """
    Visualizes the original, masked, and reconstructed images from the MAE model.
    """
    # Reshape mask
    mask = mask.detach().cpu().unsqueeze(-1).repeat(1, 1, 16**2 * 3)
    mask = unpatchify(mask)

    # Unpatchify prediction and target
    pred = pred.detach().cpu()
    pred = unpatchify(pred)
    
    # Denormalize images for visualization
    imgs = denormalize(imgs.clone())
    pred = denormalize(pred.clone())
    
    # Create masked image
    im_masked = imgs * (1 - mask)

    # Concatenate for side-by-side comparison
    imgs_all = torch.cat([imgs[:num_images], im_masked[:num_images], pred[:num_images]], dim=0)

    # Create grid and save
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2, 6))
    for i in range(num_images):
        # Original
        axes[0, i].imshow(to_pil_image(imgs[i]))
        axes[0, i].axis('off')
        # Masked
        axes[1, i].imshow(to_pil_image(im_masked[i]))
        axes[1, i].axis('off')
        # Reconstructed
        axes[2, i].imshow(to_pil_image(pred[i]))
        axes[2, i].axis('off')

    axes.set_title('Original', loc='left', fontsize=10)
    axes.set_title('Masked', loc='left', fontsize=10)
    axes.set_title('Reconstructed', loc='left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()