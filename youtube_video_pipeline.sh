#!/bin/bash

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
