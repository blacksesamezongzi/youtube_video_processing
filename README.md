## YouTube Video Frame Extraction Pipeline

This project provides a pipeline to extract frames from YouTube videos that contain specific objects: **ramp**, **zebra crossing**, and **sidewalk**.

### Overview of the Pipeline

1. **Video Downloading**  
   Reads video URLs from `video_urls.txt` and downloads the corresponding YouTube videos into the `downloaded_videos/` folder.

2. **Object Detection with OWL-V2**  
   Uses the OWL-V2 open-vocabulary detector to identify and extract frames containing the target objects. Each object type uses a different confidence threshold.

3. **Frame Filtering with BoQ**  
   Applies visual similarity filtering using the Bag-of-Queries model to reduce redundancy. Each object type uses a different similarity threshold.

---

### Prerequisites

- Clone this repository with all scripts in the root directory.
- Ensure the OWL-V2 model and Bag-of-Queries source code are downloaded and available.
- Create two environments:
  - `owlv2` for object detection (OWL-V2)
  - `boq` for visual filtering (Bag-of-Queries)

---

### Running the Pipeline

To run the full pipeline, execute the provided bash script using the following command:

```
bash youtube_video_pipeline.sh
```

For inference, the bash file is as following:

```bash
#!/bin/bash
eval "$(micromamba shell hook --shell bash)"

set -e
echo "Starting full pipeline..."

# Step 0: Download YouTube videos
echo "Downloading videos from video_urls.txt..."
python download_videos.py

# Step 1: Object Detection (OWL-V2)
echo "Activating owlv2 environment..."
micromamba activate owlv2

echo "Running object detection using OWL-V2..."
python run_owlv2.py

# Step 2: Similarity Filtering (BoQ)
echo "Switching to boq environment..."
micromamba activate boq

echo "Running similarity filtering using BoQ..."
python sparsify_frames.py

echo "Pipeline completed successfully."
