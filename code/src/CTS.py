import os
import glob
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import binary_dilation
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

def calculate_CS(primary_mask: np.ndarray,
                 secondary_mask: np.ndarray,
                 buffer_radius: int = 10,
                 threshold: float = 0.5) -> float:
    """
    Compute a generic Crack Segment Score (CS) between two binary masks.
    - primary_mask:   used to extract segments (prediction for PCS, GT for RCS)
    - secondary_mask: used as reference for overlap (GT for PCS, prediction for RCS)
    Returns a score in [0,1].
    """

    skel_p = skeletonize(primary_mask > 0)
    skel_s = skeletonize(secondary_mask > 0)


    lbl_p = label(skel_p, connectivity=2)
    lbl_s = label(skel_s, connectivity=2)


    lengths_p = {i: np.sum(lbl_p == i) for i in np.unique(lbl_p) if i > 0}
    total_p = sum(lengths_p.values())
    if total_p == 0:
        return 0.0


    selem = np.ones((2*buffer_radius+1,)*2, dtype=bool)
    buf_s = binary_dilation(skel_s, structure=selem)
    buf_p = binary_dilation(skel_p, structure=selem)

    score_sum = 0.0


    for i, Li in lengths_p.items():
        wi = Li / total_p
        seg_mask = (lbl_p == i)


        FPb = np.sum(seg_mask & ~buf_s)

        overlaps = []

        for j in [j for j in np.unique(lbl_s) if j > 0]:
            gt_j_mask = (lbl_s == j)
            TPb = np.sum(seg_mask & buf_s & gt_j_mask)
            FNb = np.sum(gt_j_mask & ~buf_p)

            corr = TPb / (TPb + FPb) if (TPb + FPb) > 0 else 0.0
            compl = TPb / (TPb + FNb) if (TPb + FNb) > 0 else 0.0


            if corr > threshold or compl > threshold:
                overlaps.append(j)

        if not overlaps:
            continue


        merged = np.zeros_like(skel_s, dtype=bool)
        for j in overlaps:
            merged |= (lbl_s == j)
        merged_buf = binary_dilation(merged, structure=selem)


        TPf = np.sum(seg_mask & merged_buf)
        FPf = np.sum(seg_mask & ~merged_buf)
        corr_f = TPf / (TPf + FPf) if (TPf + FPf) > 0 else 0.0

        if corr_f > threshold:
            score_sum += wi

    return score_sum

def compute_cts(pred_mask: np.ndarray,
                gt_mask: np.ndarray,
                buffer_radius: int = 10,
                threshold: float = 0.5) -> dict:
    """
    Compute PCS, RCS, and final Crack Topology Score (CTS).
    Returns a dict: {'PCS': ..., 'RCS': ..., 'CTS': ...}
    """
    PCS = calculate_CS(pred_mask, gt_mask, buffer_radius, threshold)
    RCS = calculate_CS(gt_mask, pred_mask, buffer_radius, threshold)
    CTS = 2 * PCS * RCS / (PCS + RCS) if (PCS + RCS) > 0 else 0.0
    return {'PCS': PCS, 'RCS': RCS, 'CTS': CTS}

def load_binary_mask(path: str) -> np.ndarray:
    """
    Load an image from disk and convert to a binary mask.
    Assumes that the input can be grayscale or RGB.
    """
    img = imread(path)
    if img.ndim == 3:

        img = rgb2gray(img)

    th = threshold_otsu(img)
    return (img > th).astype(bool)

if __name__ == "__main__":

    l = ["0_2", "2_4", "4_8", "8_16", "16_32", "thick", "zero"]

    gt_dir = f"/home/sil-juicy/Crack/Datasets/datas/split_dataset_final/split/GT/{l[5]}"


    pred_dir   = "/home/sil-juicy/Crack/save/outputs/FCN"

    buffer_radius = 10
    threshold = 0.5


    pred_paths = sorted(glob.glob(os.path.join(pred_dir, "*.*")))
    if not pred_paths:
        raise ValueError(f"No prediction files found in '{pred_dir}'.")

    results = []
    for pred_path in pred_paths:
        fname = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):

            print(f"Warning: GT not found for '{fname}', skipping.")
            continue


        pred_mask = load_binary_mask(pred_path)
        gt_mask   = load_binary_mask(gt_path)


        scores = compute_cts(pred_mask, gt_mask,
                             buffer_radius=buffer_radius,
                             threshold=threshold)
        results.append({'filename': fname, **scores})

    if not results:
        raise RuntimeError("No valid image pairs were processed.")


    pcs_mean = np.mean([r['PCS'] for r in results])
    rcs_mean = np.mean([r['RCS'] for r in results])
    cts_mean = np.mean([r['CTS'] for r in results])


    print("Per-image results:")
    for r in results:
        print(f"{r['filename']}: PCS={r['PCS']:.3f}, "
              f"RCS={r['RCS']:.3f}, CTS={r['CTS']:.3f}")


    print("\nAverage over all images:")
    print(f"Mean PCS={pcs_mean:.3f}, Mean RCS={rcs_mean:.3f}, Mean CTS={cts_mean:.3f}")
