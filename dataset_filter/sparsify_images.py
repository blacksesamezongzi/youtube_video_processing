import os
import sys
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, Manager
import torchvision.transforms as T  # Import torchvision.transforms

'''
Usage:
python sparsify_images.py \
  --input-folder "/home/clearlab/youtube_video_processing/20250908_134025" \
  --output-folder "/home/clearlab/youtube_video_processing/20250908_134025_boq" \
  --object-type "sidewalk" \
  --similarity-threshold 0.5 \
  --num-workers 8
'''

# Add path to BoQ
sys.path.insert(0, os.path.abspath("Bag-of-Queries/src"))
from boq import BoQ

# Global variable for the model
model = None

def load_image_tensor(path):
    """Load an image and convert it to a tensor."""
    img = Image.open(path).convert("RGB")
    tensor = T.ToTensor()(img).unsqueeze(0)  # Add batch dimension
    return img, tensor

def sparsify_images_worker(args):
    """Worker function for sparsifying images."""
    input_path, saved_embeddings, saved_filenames, output_folder, similarity_threshold = args

    try:
        pil_img, tensor_img = load_image_tensor(input_path)

        with torch.no_grad():
            emb = model.get_embedding(tensor_img.to(model.device))  # Ensure tensor is on the correct device
            emb = emb / emb.norm(p=2)  # Normalize the embedding
            emb = emb.cpu()  # Move the embedding to the CPU for comparison

        # Compare the current image embedding with all saved embeddings
        is_similar = any(torch.dot(emb, e).item() >= similarity_threshold for e in saved_embeddings)

        if not is_similar:
            saved_embeddings.append(emb)  # Append the CPU tensor to the shared list
            saved_filenames.append(input_path)
            output_path = os.path.join(output_folder, os.path.basename(input_path))
            pil_img.save(output_path)
            print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error processing {input_path}: {e}")

def init_worker():
    """Initialize the BoQ model for each worker."""
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BoQ(backbone_name="resnet50", device=device)

def parse_args():
    parser = argparse.ArgumentParser(description="Sparsify images based on similarity")
    parser.add_argument("--input-folder", required=True, help="Path to the input folder containing images")
    parser.add_argument("--output-folder", required=True, help="Path to save the sparsified images")
    parser.add_argument("--object-type", required=True, choices=["sidewalk", "zebra_crossing"], help="Object type to process")
    parser.add_argument("--similarity-threshold", type=float, default=0.5, help="Similarity threshold for sparsification")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    object_type = args.object_type
    similarity_threshold = args.similarity_threshold
    num_workers = args.num_workers

    # Adjust thresholds based on object type
    thresholds = {
        "sidewalk": 0.5,
        "zebra_crossing": 0.6,
    }
    similarity_threshold = thresholds.get(object_type, similarity_threshold)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Recursively find all image files
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    image_files = []
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.lower().endswith(image_extensions) and not f.startswith("."):
                image_files.append(os.path.join(root, f))
    image_files = sorted(image_files)

    if not image_files:
        print("⚠️ No images found in the input folder.")
        exit(0)

    print(f"📦 Found {len(image_files)} images. Starting sparsification with {num_workers} workers...")

    # Shared lists for embeddings and filenames
    manager = Manager()
    saved_embeddings = manager.list()
    saved_filenames = manager.list()

    # Use multiprocessing to process images in parallel
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        pool.map(
            sparsify_images_worker,
            [(image, saved_embeddings, saved_filenames, output_folder, similarity_threshold) for image in image_files]
        )

    print(f"🎉 Sparsification complete. Saved images to {output_folder}")