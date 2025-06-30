import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff
import open3d as o3d
import os

def load_images(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return grayA, grayB


def compute_ssim(img1, img2):
    return ssim(img1, img2, full=True)


def compute_iou(img1, img2, threshold=0.5):
    bin1 = (img1 < threshold).astype(np.uint8)
    bin2 = (img2 < threshold).astype(np.uint8)

    intersection = np.logical_and(bin1, bin2).sum()
    union = np.logical_or(bin1, bin2).sum()

    return intersection / union if union > 0 else 0


def compute_adjustment_error(gt_map, slam_map, threshold=0.5):
    """
    Compute Hausdorff distance-based adjustment error between 
    a ground truth map and a SLAM-generated map.
    
    Parameters:
        gt_map (ndarray): Ground truth map (grayscale, float or uint8)
        slam_map (ndarray): SLAM-generated map (grayscale, same size/type as gt_map)
        threshold (float): Threshold for binarization (0–1 for float images, 0–255 for uint8)
        
    Returns:
        float: Symmetric Hausdorff distance (max of directed distances)
    """

    # Ensure the maps are in compatible dtype
    if gt_map.dtype != np.uint8:
        gt_bin = (gt_map < threshold).astype(np.uint8) * 255
        slam_bin = (slam_map < threshold).astype(np.uint8) * 255
    else:
        gt_bin = (gt_map < threshold).astype(np.uint8) * 255
        slam_bin = (slam_map < threshold).astype(np.uint8) * 255

    # Extract contours
    contours_gt, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_slam, _ = cv2.findContours(slam_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Handle empty contour cases
    if not contours_gt or not contours_slam:
        return float('inf')

    pts_gt = np.vstack(contours_gt).squeeze()
    pts_slam = np.vstack(contours_slam).squeeze()

    if pts_gt.ndim < 2 or pts_slam.ndim < 2:
        return float('inf')

    # Calculate symmetric Hausdorff distance
    hd1 = directed_hausdorff(pts_gt, pts_slam)[0]
    hd2 = directed_hausdorff(pts_slam, pts_gt)[0]

    return max(hd1, hd2)

def image_to_point_cloud(binary_img):
    # Extract (y,x) coordinates of black (occupied) pixels
    points = np.column_stack(np.where(binary_img == 0))
    # Convert to (x,y) float format
    return points.astype(np.float32)

def compute_icp_error(map1_img, map2_img, resolution=1.0):
    # Convert to binary
    _, bin1 = cv2.threshold(map1_img, 127, 255, cv2.THRESH_BINARY)
    _, bin2 = cv2.threshold(map2_img, 127, 255, cv2.THRESH_BINARY)

    # Convert to point clouds
    pts1 = image_to_point_cloud(bin1)
    pts2 = image_to_point_cloud(bin2)

    # Convert to Open3D format
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(np.c_[pts1[:, 1], -pts1[:, 0], np.zeros(len(pts1))] * resolution)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(np.c_[pts2[:, 1], -pts2[:, 0], np.zeros(len(pts2))] * resolution)

    # ICP registration
    threshold = 2.0  # max correspondence distance (in meters)
    result = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Transform pcd1 using ICP result
    aligned = pcd1.transform(result.transformation)

    # Compute distances from aligned pcd1 to pcd2
    distances = np.asarray(aligned.compute_point_cloud_distance(pcd2))

    stats = {
        "mean": np.mean(distances),
        "rmse": np.sqrt(np.mean(distances**2))
    }

    print("ICP Error Metrics:")
    print(f"  Mean Error : {stats['mean']:.4f}")
    print(f"  RMSE       : {stats['rmse']:.4f}")


if __name__ == "__main__":
    # === Replace these paths ===
    #gmapping, slam_toolbox, lidar_rtabmap, rgbd_rtabmap, cartographer
    path1 = "/home/jh/worlds-magisterka/test_results/warehouse/cartographer_map_cropped.pgm"
    print(path1)
    path2 = "/home/jh/worlds-magisterka/ground_truths/warehouse_ground_truth.pgm"

    if not (os.path.exists(path1) and os.path.exists(path2)):
        print("Error: PGM files not found.")
        exit(1)

    slam_map, gt_map = load_images(path1, path2)



    # Compute metrics
    (score, diff) = compute_ssim(slam_map, gt_map)
    print(f"SSIM:           {score:.4f}")
    print(f"IoU:            {compute_iou(slam_map, gt_map):.4f}")
    print(f"Hausdorff Dist: {compute_adjustment_error(gt_map, slam_map):.2f}")
    compute_icp_error(slam_map, gt_map)

