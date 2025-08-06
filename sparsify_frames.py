import os
import sys
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# Add path to BoQ
sys.path.insert(0, os.path.abspath("Bag-of-Queries/src"))
from boq import BoQ

def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    tensor = T.ToTensor()(img).permute(1, 2, 0)
    return img, tensor

def sparsify_folder(input_folder, output_folder, similarity_threshold):
    os.makedirs(output_folder, exist_ok=True)

    saved_embeddings = []
    saved_filenames = []

    image_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".png"))
    ])

    for filename in tqdm(image_files, desc=f"→ {os.path.basename(input_folder)}"):
        img_path = os.path.join(input_folder, filename)
        pil_img, tensor_img = load_image_tensor(img_path)

        with torch.no_grad():
            emb = model.get_embedding(tensor_img)
            emb = emb / emb.norm(p=2)

        is_similar = any(torch.dot(emb, e).item() >= similarity_threshold for e in saved_embeddings)

        if not is_similar:
            saved_embeddings.append(emb)
            saved_filenames.append(filename)
            pil_img.save(os.path.join(output_folder, filename))

if __name__ == "__main__":
    input_root = "object_detector_output"
    output_root = "similarity_filter_output"
    os.makedirs(output_root, exist_ok=True)

    thresholds = {
        "sidewalk": 0.5,
        "ramp": 0.6,
        "zebra_crossing": 0.5
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BoQ(backbone_name="resnet50", device=device)

    for video_name in sorted(os.listdir(input_root)):
        video_path = os.path.join(input_root, video_name)
        if not os.path.isdir(video_path):
            continue

        for object_type, threshold in thresholds.items():
            class_name = object_type.replace(" ", "_").lower()
            input_dir = os.path.join(video_path, class_name)

            if not os.path.isdir(input_dir):
                continue

            output_dir = os.path.join(output_root, class_name, video_name)
            print(f"{video_name} | {class_name}")
            sparsify_folder(input_dir, output_dir, threshold)
