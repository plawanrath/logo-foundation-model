# @plawanrath

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def collate_local_images(source_dir, output_dir, prefix):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    if not source_path.is_dir():
        print(f"ERROR: Source directory not found: {source_path}")
        return 0

    print(f"Collating images from {source_path}")
    files = [p for p in source_path.glob('**/*') if p.suffix.lower() in image_extensions]
    total = 0
    used_names = set()
    
    for p in tqdm(files, desc=f"Copying {prefix}"):
        # Create unique filename to avoid overwrites
        base_name = f"{prefix}_{p.stem}"
        dest_name = f"{base_name}{p.suffix}"
        counter = 1
        
        # If name already exists, add counter
        while dest_name in used_names:
            dest_name = f"{base_name}_{counter}{p.suffix}"
            counter += 1
        
        used_names.add(dest_name)
        dest = output_path / dest_name
        shutil.copy2(p, dest)
        total += 1
    return total

if __name__ == '__main__':
    logo2k_path = "./datasets/LogoDet-2K"      
    outdir = "data/pretrain/images"

    c = collate_local_images(logo2k_path, outdir, "local_Logo2Kplus")

    print("\nDone.")
    print(f"Total images copied: {c}")
