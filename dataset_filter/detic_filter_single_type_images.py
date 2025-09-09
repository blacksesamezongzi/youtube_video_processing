#!/usr/bin/env python3
"""
Usage:
python detic_filter_single_type_images.py \
  --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
  --weights models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
  --input-root /home/clearlab/youtube_video_processing/20250908_134025 \
  --output-folder /home/clearlab/youtube_video_processing/20250908_filtered \
  --target-class "zebra crossing" \
  --confidence-threshold 0.1
"""

import os
import sys
from pathlib import Path
import argparse
import shutil

# Make sure CenterNet2 is on sys.path (matches your repo layout)
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_DIR, "third_party", "CenterNet2"))

import torch
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

CANONICAL_CLASSES = {
    "sidewalk": "sidewalk",
    "zebra crossing": "zebra crossing",
    "zebra_crossing": "zebra crossing",
}

def build_cfg(config_file: str, weights: str, score_thresh: float, device: str):
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    return cfg

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def normalize_target_class(s: str) -> str:
    key = s.strip().lower().replace("-", " ")
    return CANONICAL_CLASSES.get(key, None)

def main():
    parser = argparse.ArgumentParser(description="Filter images by a single Detic class and save to a single folder.")
    parser.add_argument("--input-root", required=True, help="Root folder of input images (recursively searched).")
    parser.add_argument("--output-folder", required=True, help="Folder where kept images will be saved.")
    parser.add_argument("--target-class", required=True, choices=["sidewalk", "zebra crossing", "zebra_crossing"],
                        help="Target class to keep (others are ignored).")
    parser.add_argument("--config-file", required=False,
                        default="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    parser.add_argument("--weights", required=False,
                        default="models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
    parser.add_argument("--confidence-threshold", type=float, default=0.40,
                        help="Confidence threshold for the target class.")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Debug: stop after processing N images (0 = all).")
    args = parser.parse_args()

    setup_logger()

    # Canonicalize class string (and validate)
    target_class = normalize_target_class(args.target_class)
    if target_class is None:
        print(f"[ERROR] Unsupported target class: {args.target_class}", file=sys.stderr)
        sys.exit(1)

    input_root = Path(args.input_root).resolve()
    output_folder = Path(args.output_folder).resolve()
    if not input_root.exists() or not input_root.is_dir():
        print(f"[ERROR] Input root not found or not a directory: {input_root}", file=sys.stderr)
        sys.exit(1)

    # Device & config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = build_cfg(args.config_file, args.weights, args.confidence_threshold, device)

    # VisualizationDemo with custom vocabulary = only the target class
    detic_args = argparse.Namespace(
        vocabulary="custom",
        custom_vocabulary=target_class,
    )
    demo = VisualizationDemo(cfg, detic_args)

    # Collect all images recursively
    all_images = [p for p in input_root.rglob("*") if p.is_file() and is_image_file(p)]
    all_images.sort()
    if not all_images:
        print(f"[WARN] No images found under {input_root}")
        sys.exit(0)

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    scanned = 0
    kept = 0

    for i, img_path in enumerate(tqdm(all_images, desc=f"Scanning for '{target_class}'")):
        if args.max_images and i >= args.max_images:
            break
        scanned += 1
        try:
            # Detectron2 expects BGR
            image = read_image(str(img_path), format="BGR")
            predictions, _ = demo.run_on_image(image)
            instances = predictions.get("instances", None)
            if instances is None or len(instances) == 0:
                continue

            inst = instances.to("cpu")
            scores = inst.scores.tolist()
            classes = inst.pred_classes.tolist()

            # Since vocabulary has ONLY the target class, its index is 0
            max_score = 0.0
            for c, s in zip(classes, scores):
                if c == 0:
                    if s > max_score:
                        max_score = float(s)

            if max_score >= args.confidence_threshold:
                out_path = output_folder / img_path.name
                shutil.copy2(img_path, out_path)
                kept += 1
        except Exception as e:
            print(f"[WARN] Skipping {img_path}: {e}", file=sys.stderr)

    print(f"\nDone. Target class: {target_class}")
    print(f"Scanned: {scanned} images | Kept: {kept} | Output folder: {output_folder}")

if __name__ == "__main__":
    main()

