import cv2
import os
import numpy as np

def find_image_pairs(input_dir):
    files = os.listdir(input_dir)
    ids = set(f.split('_')[0] for f in files if f.endswith('.JPG'))
    pairs = [(os.path.join(input_dir, f"{id}_T.JPG"), os.path.join(input_dir, f"{id}_Z.JPG")) for id in ids]
    return pairs

def align_images(thermal, rgb):
    thermal_gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(thermal_gray, None)
    kp2, des2 = orb.detectAndCompute(rgb_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 4:
        print("Not enough matches to align")
        return thermal
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    matrix, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    aligned = cv2.warpPerspective(thermal, matrix, (rgb.shape[1], rgb.shape[0]))
    return aligned

def overlay_images(rgb, aligned_thermal):
    thermal_colored = cv2.applyColorMap(aligned_thermal, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb, 0.6, thermal_colored, 0.4, 0)
    return overlay

def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pairs = find_image_pairs(input_dir)
    for thermal_path, rgb_path in pairs:
        id = os.path.basename(thermal_path).split('_')[0]
        thermal = cv2.imread(thermal_path)
        rgb = cv2.imread(rgb_path)
        aligned_thermal = align_images(thermal, rgb)
        thermal_gray = cv2.cvtColor(aligned_thermal, cv2.COLOR_BGR2GRAY)
        overlay = overlay_images(rgb, thermal_gray)
        out_path = os.path.join(output_dir, f"{id}_overlay.jpg")
        cv2.imwrite(out_path, overlay)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"
    process_images(input_folder, output_folder)
