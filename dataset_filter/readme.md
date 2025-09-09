## Script Usage & Environment Setup

### Environment Setup

- [**OWL-V2**](https://huggingface.co/docs/transformers/model_doc/owlv2): Create a dedicated environment (e.g., `owlv2`) for running OWL-V2 object detection scripts.  

- [**BoQ**](https://github.com/amaralibey/Bag-of-Queries): Create a separate environment (e.g., `boq`) for Bag-of-Queries similarity filtering.  
  **Important:** Replace the original `boq.py` in the BoQ source with the version provided in `dataset_filter/env/boq.py`.
- [**Detic**](https://github.com/facebookresearch/Detic): Create an environment (e.g., `detic`) for Detic-based filtering.

### Folder Structure & Placement

- Place `run_owlv2_images.py` and `sparsify_images.py` inside the root folder.
- Place `detic_filter_images.py` inside the `Detic` folder.
- Before running `sparsify_images.py`, `cd` into the `Detic` directory.

### Usage Reminders

1. **Configure Thresholds:**  
   Before running the scripts on the full dataset, test with a small subset (e.g., 200 images) to determine appropriate confidence and similarity thresholds for your use case.

2. **Execution Example**
   Execution commands for each script are written right after import statements in the respective Python files.

**Tip:** Always verify output quality and adjust thresholds before processing the huge dataset.