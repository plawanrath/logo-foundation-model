# @plawanrath

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
import cv2
import random

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
    h = w = int(x.shape[1]**0.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
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

    axes[0, 0].set_title('Original', fontsize=12)
    axes[1, 0].set_title('Masked', fontsize=12)
    axes[2, 0].set_title('Reconstructed', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_coco_api_from_dataset(dataset):
    # This is a bit of a workaround to get the COCO API object
    # from the custom dataset, which is needed to get catIds.
    return dataset.coco

def visualize_predictions(image_path, predictions, id_to_label, save_path):
    """
    Draws predicted bounding boxes on an image.
    """
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Generate a color for each class
    colors = {}
    for label_id in id_to_label.keys():
        colors[label_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for score, label_id, box in zip(predictions['scores'], predictions['labels'], predictions['boxes']):
        box = [int(i) for i in box.tolist()]
        label = id_to_label.get(label_id.item(), 'Unknown')
        color = colors.get(label_id.item(), (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img, (box, box), (box, box), color, 2)
        
        # Draw label
        text = f"{label}: {score:.2f}"
        cv2.putText(img, text, (box, box - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Inference visualization saved to {save_path}")