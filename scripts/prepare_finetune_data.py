# @plawanrath
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil

def parse_voc_annotation(xml_path, label_map):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        if cls not in label_map:
            continue
        cid = label_map[cls]
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append((cid, xmin, ymin, xmax, ymax))
    return bboxes

def convert_detection_to_coco(det_dir, output_dir, train_ratio=0.8):
    det_dir = Path(det_dir)
    out = Path(output_dir)
    categories = []
    label_map = {}
    ann_id = 0

    # Collect all brand names across all categories
    all_brands = set()
    all_files = []
    
    for category_dir in det_dir.iterdir():
        if not category_dir.is_dir():
            continue
        for brand_dir in category_dir.iterdir():
            if not brand_dir.is_dir():
                continue
            brand_name = brand_dir.name
            all_brands.add(brand_name)
            
            # Collect all image/xml pairs
            for img_file in brand_dir.glob("*.jpg"):
                xml_file = brand_dir / f"{img_file.stem}.xml"
                if xml_file.exists():
                    all_files.append((img_file, xml_file, brand_name))

    # Create categories from all brands
    classes = sorted(all_brands)
    for idx, cls in enumerate(classes):
        label_map[cls] = idx
        categories.append({"id": idx, "name": cls, "supercategory": "logo"})

    # Split files into train/val
    import random
    random.seed(42)  # for reproducible splits
    random.shuffle(all_files)
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    splits = {"train": train_files, "val": val_files}

    for split_name, files in splits.items():
        coco = {"info": {"description": "LogoDet-3K to COCO"},
                "images": [], "annotations": [], "categories": categories}
        img_out = out / split_name / "images"
        img_out.mkdir(parents=True, exist_ok=True)

        for image_id, (img_path, xml_path, brand_name) in enumerate(tqdm(files, desc=f"{split_name} images")):
            bboxes = parse_voc_annotation(xml_path, label_map)
            if not bboxes:
                continue
                
            # Copy image with sequential naming
            dest = img_out / f"{image_id:012d}{img_path.suffix}"
            shutil.copy2(img_path, dest)
            
            im = Image.open(img_path)
            coco["images"].append({"id": image_id, "file_name": dest.name,
                                   "width": im.width, "height": im.height})
            
            for cid, xmin, ymin, xmax, ymax in bboxes:
                w = xmax - xmin
                h = ymax - ymin
                coco["annotations"].append({
                    "id": ann_id, "image_id": image_id,
                    "category_id": cid, "bbox": [xmin, ymin, w, h],
                    "area": float(w * h), "iscrowd": 0, "segmentation": []
                })
                ann_id += 1

        with open(out / split_name / "annotations.json", "w") as f:
            json.dump(coco, f, indent=4)
            
    print(f"COCO conversion done. Train: {len(train_files)}, Val: {len(val_files)}, Classes: {len(classes)}")

if __name__ == '__main__':
    local3k = "./datasets/LogoDet-3K"   # root of your Kaggle extracted LogoDet‑3K
    outcoco = "data/finetune"
    convert_detection_to_coco(local3k, outcoco)
