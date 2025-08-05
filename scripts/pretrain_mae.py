# @plawanrath

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import argparse
from pathlib import Path

from src.datasets import UnlabeledImageDataset
from src.models import MaskedAutoencoderViT
from src.utils import plot_loss_curve, visualize_mae_reconstruction

def main(args):
    # --- Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # --- Data ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = UnlabeledImageDataset(root_dir=args.data_dir, transform=transform)

    # Create validation split
    val_size = min(1000, int(0.05 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Training with {len(train_dataset)} images, validating with {len(val_dataset)} images.")

    # --- Model & Optimizer ---
    model = MaskedAutoencoderViT(model_name='vit_small_patch16_224').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    # --- Training Loop ---
    train_losses = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for i, imgs in enumerate(train_loader):
            # UnlabeledImageDataset returns a tuple (img,)
            imgs = imgs[0].to(device)

            optimizer.zero_grad()
            loss, _, _ = model(imgs)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f"--- End of Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_epoch_loss:.4f} ---")

        # --- Checkpoint & Visualize ---
        if (epoch + 1) % args.save_interval == 0:
            encoder_weights_path = Path(args.checkpoint_dir) / f"mae_vit_encoder_epoch_{epoch+1}.pth"
            torch.save(model.encoder.state_dict(), encoder_weights_path)
            print(f"Saved encoder weights to {encoder_weights_path}")

            # Visualize reconstruction
            model.eval()
            val_imgs = next(iter(val_loader))[0].to(device)  # unpack batch
            with torch.no_grad():
                _, pred, mask = model(val_imgs)

            vis_path = Path(args.output_dir) / f"reconstruction_epoch_{epoch+1}.png"
            visualize_mae_reconstruction(val_imgs.cpu(), pred.cpu(), mask.cpu(), vis_path)
            print(f"Saved reconstruction visualization to {vis_path}")

    # --- Final ---
    plot_loss_curve(train_losses, Path(args.output_dir) / "pretrain_loss_curve.png")
    print("✅ Pre-training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAE Pre-training Script')
    parser.add_argument('--data_dir', type=str, default='data/pretrain/images', help='Directory of pre-training images')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/mae_pretrained', help='Directory to save model checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save visualizations and plots')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='Learning rate')
    parser.add_argument('--save_interval', type=int, default=10, help='Epoch interval to save checkpoints and visualizations')

    args = parser.parse_args()
    main(args)
