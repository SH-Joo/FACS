import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.morphology import disk, thin

def binarize(img):
    """
    Adaptive binarization:
    - If only two values (e.g. 0,255), treat the less frequent as foreground
    - Otherwise apply Otsu thresholding
    """
    vals, counts = np.unique(img, return_counts=True)
    if len(vals) == 2:
        fg = vals[np.argmin(counts)]
        return (img == fg).astype(np.uint8)
    else:
        _, thr = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thr

def apply_tolerance(true, pred, tol):
    """
    Dilate true/pred by disk(tol) to allow tolerance, then compute new masks.
    """
    true_dil = cv2.dilate(true, disk(tol), iterations=1)
    pred_dil = cv2.dilate(pred, disk(tol), iterations=1)

    tp = true * pred_dil
    fp = pred - (pred * true_dil)
    fn = true - tp

    true_new = tp + fn
    pred_new = tp + fp
    return true_new, pred_new

def compute_iou_manual(true, pred):
    """
    Compute IoU; if union==0 return 1.0
    """
    intersection = np.logical_and(true, pred).sum()
    union = np.logical_or(true, pred).sum()
    if union == 0:
        return 1.0
    return intersection / union

def run_evaluation(gt_folder, pred_folder, tolerances):
    gt_folder = Path(gt_folder)
    pred_folder = Path(pred_folder)

    gt_files = sorted(gt_folder.glob("*.png"))
    matched_pairs = []
    for gt_file in gt_files:
        stem = gt_file.stem
        pred_file = pred_folder / f"{stem}_pred.png"
        if pred_file.exists():
            matched_pairs.append((gt_file, pred_file))
        else:
            print(f"[!] Prediction not found for {stem}")

    if not matched_pairs:
        print("No matching files.")
        return

    # Prepare containers for cl-IoU and raw IoU
    all_trues = {tol: [] for tol in tolerances}
    all_preds = {tol: [] for tol in tolerances}
    iou_list = []  # for storing raw IoU per image

    total = len(matched_pairs)
    for idx, (gt_path, pred_path) in enumerate(matched_pairs, 1):
        print(f"[{idx:3d}/{total}] Processing {gt_path.name}...")

        true_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

        # adaptive binarization
        true_bin = binarize(true_img)
        pred_bin = binarize(pred_img)

        # Raw IoU 계산 및 저장
        raw_iou = compute_iou_manual(true_bin.flatten(), pred_bin.flatten())
        iou_list.append(raw_iou)

        # thinning to skeletons
        true_skel = thin(true_bin.astype(bool)).astype(np.uint8)
        pred_skel = thin(pred_bin.astype(bool)).astype(np.uint8)

        # resize if shapes mismatch
        if pred_skel.shape != true_skel.shape:
            pred_skel = cv2.resize(
                pred_skel,
                (true_skel.shape[1], true_skel.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # cl‑IoU 저장
        for tol in tolerances:
            t_tol, p_tol = apply_tolerance(true_skel, pred_skel, tol)
            ft = t_tol.flatten()
            fp = p_tol.flatten()
            keep = np.where((ft == 1) | (fp == 1))[0]
            all_trues[tol].append(ft[keep])
            all_preds[tol].append(fp[keep])

    # cl‑IoU 출력
    print("\n=== cl-IoU Results ===")
    for tol in tolerances:
        tr = np.concatenate(all_trues[tol])
        pr = np.concatenate(all_preds[tol])
        score = compute_iou_manual(tr, pr)
        print(f"Tolerance {tol:2d}: IoU = {score:.4f}")

    # 평균 raw IoU 출력
    mean_raw_iou = np.mean(iou_list) if iou_list else float('nan')
    print(f"\n=== Raw IoU ===\nMean IoU over all images: {mean_raw_iou:.4f}")

if __name__ == "__main__":
    # gt_path = "/home/sil-juicy/Crack/Datasets/datas/split_dataset_final/test/GT"
    gt_path = "/home/sil-juicy/Crack/Datasets/datas/split_dataset_final/split/GT/0_2"
    pred_path = "/home/sil-juicy/Downloads/v1"
    # pred_path = "/home/sil-juicy/codes/Segmentation/outputs/U-Net/predictions/v1"
    
    tolerance_list = [0, 1, 2, 4, 8, 16, 32, 64]

    run_evaluation(gt_path, pred_path, tolerance_list)
