### Subsequent Training Steps

1. SSH into the remote server `sam2`.

2. In `/home/spatial/sam2/sam2/sam2/configs/final_configs`, locate the file `tiny_NoneNone.yaml`.

    - Copy and rename it within the same folder.

    - Note the new filename.

    - Edit the file to update:

        - Line 16: img_folder → path to the input images.

        - Line 17: gt_folder → path to the ground truth labels.

3. In `/home/spatial/sam2/sam2/training`, locate the file `ttt_NoneNone.py`.

    - Copy and rename it within the same folder.

    - On line 249, update the default parameter to the path of the YAML file created in Step 2.

4. Activate the `sam2` environment, navigate to `/home/spatial/sam2/sam2/training`, and run the Python script.

**Pending clarification**: Output format and location are currently unknown.