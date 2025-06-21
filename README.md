# Thermal-RGB Image Overlay Assignment

## Objective
To align thermal images with their RGB counterparts and generate overlaid visualizations using feature-based matching.

## How It Works
1. Load all image pairs with `_T.JPG` (thermal) and `_Z.JPG` (RGB) naming.
2. Align thermal image to RGB using ORB features + homography.
3. Overlay the thermal (converted to heatmap) on RGB using `cv2.addWeighted`.
4. Save outputs to `/output` directory.

## Usage
```bash
python overlay_script.py
