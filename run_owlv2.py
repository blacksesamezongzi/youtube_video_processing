from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import torch
import cv2
import os

def process_video_with_owlv2(video_path, output_root="object_detector_output"):
    prompts = ["sidewalk", "ramp", "zebra crossing"]
    thresholds = {
        "sidewalk": 0.07,
        "ramp": 0.03,
        "zebra crossing": 0.05
    }

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_root, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    class_to_folder = {}
    for cls in prompts:
        folder_name = cls.replace(" ", "_").lower()
        folder_path = os.path.join(video_output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        class_to_folder[cls] = folder_path

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // 3) if fps else 1
    frame_id = 0

    print(f"🎥 Processing {video_path} ({fps:.2f} fps)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            inputs = processor(images=pil_image, text=prompts, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            target_size = torch.tensor([pil_image.size[::-1]])
            results = processor.post_process(outputs, target_sizes=target_size)[0]

            for label_id, score in zip(results["labels"], results["scores"]):
                class_name = prompts[label_id]
                threshold = thresholds[class_name]
                if score.item() > threshold:
                    folder = class_to_folder[class_name]
                    save_path = os.path.join(folder, f"frame_{frame_id:04d}.jpg")
                    if not os.path.exists(save_path):
                        pil_image.save(save_path)
                        print(f"Saved: {save_path}")

        frame_id += 1

    cap.release()
    print(f"Finished {video_path}")


if __name__ == "__main__":
    import sys
    from transformers import logging
    logging.set_verbosity_error()  # hide warnings

    input_folder = "downloaded_videos"
    if not os.path.exists(input_folder):
        print(f"Folder '{input_folder}' not found.")
        sys.exit(1)

    print("Loading OWL-V2 model...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    video_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".mp4")
    ])

    if not video_files:
        print("No videos found in 'downloaded_videos/'.")
        sys.exit(0)

    for filename in video_files:
        video_path = os.path.join(input_folder, filename)
        process_video_with_owlv2(video_path)

