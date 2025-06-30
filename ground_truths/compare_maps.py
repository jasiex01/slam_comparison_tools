import cv2
import numpy as np
import argparse

def load_binary_map(path):
    """Load a PGM map and threshold it to binary."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image from {path}")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary

def resize_to_match(src, target_shape):
    """Resize src to match target shape."""
    return cv2.resize(src, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

def compute_iou(gt, slam):
    intersection = cv2.bitwise_and(gt, slam)
    union = cv2.bitwise_or(gt, slam)
    inter_count = cv2.countNonZero(intersection)
    union_count = cv2.countNonZero(union)
    iou = inter_count / union_count if union_count > 0 else 0
    return iou

def compute_pixel_accuracy(gt, slam):
    return np.sum(gt == slam) / gt.size

def main(gt_path, slam_path, diff_out):
    print("Loading and processing maps...")

    gt_map = load_binary_map(gt_path)
    slam_map = load_binary_map(slam_path)

    if gt_map.shape != slam_map.shape:
        print("Resizing SLAM map to match ground truth size...")
        slam_map = resize_to_match(slam_map, gt_map.shape)

    print("Computing metrics...")
    iou = compute_iou(gt_map, slam_map)
    accuracy = compute_pixel_accuracy(gt_map, slam_map)

    print(f"\n=== Map Comparison Results ===")
    print(f"IoU (Intersection over Union): {iou:.4f}")
    print(f"Pixel-wise Accuracy:          {accuracy:.4f}")

    # Save diff image
    diff = cv2.absdiff(gt_map, slam_map)
    cv2.imwrite(diff_out, diff)
    print(f"Difference image saved to: {diff_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SLAM map with ground truth")
    parser.add_argument("--gt", required=True, help="Path to ground truth PGM map")
    parser.add_argument("--slam", required=True, help="Path to SLAM-generated PGM map")
    parser.add_argument("--out", default="map_diff.png", help="Output path for difference image")

    args = parser.parse_args()
    main(args.gt, args.slam, args.out)
