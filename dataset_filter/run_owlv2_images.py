from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import torch
import cv2
import os
import argparse
import sys
from multiprocessing import Pool, cpu_count
from transformers import logging

# input "zebra_crossing" or "sidewalk" for object-type
'''
python run_owlv2_parallel_images.py \
  --input-folder "/home/clearlab/youtube_video_processing/20250908_134025" \
  --output-folder "/home/clearlab/youtube_video_processing/20250908_output" \
  --object-type "sidewalk" \
  --num-workers 8
'''

# Global variables to be initialized once per process
model = None
processor = None
device = None

def init_model():
    global model, processor, device
    print("Initializing OWL-ViT model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    model.eval()
    print("✅ Model and processor initialized successfully.")

def process_image_with_owlv2(image_path, output_root="object_detector_output"):
    global model, processor, device

    try:
        # Only one object type per run
        object_type = process_image_with_owlv2.object_type
        prompt = "sidewalk" if object_type == "sidewalk" else "zebra crossing"
        threshold = 0.1 if object_type == "sidewalk" else 0.07

        # Generate the output path directly in the root folder
        image_name = os.path.basename(image_path)  # Keep only the file name
        save_path = os.path.join(output_root, image_name)

        pil_image = Image.open(image_path).convert("RGB")
        inputs = processor(images=pil_image, text=[prompt], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_size = torch.tensor([pil_image.size[::-1]]).to(device)
        results = processor.post_process(outputs, target_sizes=target_size)[0]

        # Save the image if it meets the threshold
        for label_id, score in zip(results["labels"], results["scores"]):
            if score.item() > threshold:
                if not os.path.exists(save_path):
                    pil_image.save(save_path)
                    print(f"✅ Saved: {save_path}")
                break  # Save the image only once if it meets the threshold
        print(f"✅ Finished {image_path}")

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run OWL-ViT for single object detection (sidewalk or zebra crossing) on images in parallel")
    parser.add_argument("--input-folder", required=True, help="Path to the input folder containing images (recursively)")
    parser.add_argument("--output-folder", default="object_detector_output", help="Path to save the output")
    parser.add_argument("--object-type", required=True, choices=["sidewalk", "zebra_crossing"], help="Object type to detect")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel processes to use")
    return parser.parse_args()

def process_image_wrapper(args):
    """Wrapper function for multiprocessing."""
    image_path, output_root = args
    process_image_with_owlv2(image_path, output_root)

if __name__ == "__main__":
    args = parse_args()
    logging.set_verbosity_error()

    input_folder = args.input_folder
    output_folder = args.output_folder
    object_type = args.object_type
    num_workers = args.num_workers

    # Set object type for detection
    process_image_with_owlv2.object_type = object_type

    if not os.path.exists(input_folder):
        print(f"❌ Folder '{input_folder}' not found.")
        sys.exit(1)

    # Recursively find all image files
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    image_files = []
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.lower().endswith(image_extensions) and not f.startswith("."):
                image_files.append(os.path.join(root, f))
    image_files = sorted(image_files)
    print(f"DEBUG: Found {len(image_files)} images: {image_files}")

    if not image_files:
        print("⚠️ No images found in the input folder.")
        sys.exit(0)

    print(f"📦 Found {len(image_files)} images. Starting processing with {num_workers} workers...")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Use multiprocessing to process images in parallel
    with Pool(processes=num_workers, initializer=init_model) as pool:
        pool.map(process_image_wrapper, [(image, output_folder) for image in image_files])
