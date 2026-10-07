"""Binary-mask IoU, CL-IoU, and CTS with explicit aggregation policies."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter
from skimage.measure import label
from skimage.morphology import skeletonize, thin


PROTOCOL = "facs-legacy-compatible-metrics-v1"


def masks(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("Metrics require equal 2D masks")
    for mask in (prediction, target):
        if not np.isfinite(mask).all() or not set(np.unique(mask).tolist()) <= {0, 1, False, True}:
            raise ValueError("Metrics consume binary masks, not logits or grayscale images")
    return prediction.astype(bool), target.astype(bool)


def ratio(numerator, denominator, *, empty_value):
    return numerator/denominator if denominator else float(empty_value)


def centerline_counts(pred_skeleton, gt_skeleton, delta):
    """Omni tolerance: TP=GT within delta of P, FP=P outside delta of GT."""
    if not isinstance(delta, int) or delta < 0:
        raise ValueError("CL-IoU tolerance must be a nonnegative integer pixel radius")
    if pred_skeleton.any():
        gt_matched = gt_skeleton & (distance_transform_edt(~pred_skeleton) <= delta)
    else:
        gt_matched = np.zeros_like(gt_skeleton)
    if gt_skeleton.any():
        false_positive = pred_skeleton & (distance_transform_edt(~gt_skeleton) > delta)
    else:
        false_positive = pred_skeleton
    tp, fp = int(gt_matched.sum()), int(false_positive.sum())
    fn = int(gt_skeleton.sum())-tp
    return {"tp": tp, "fp": fp, "fn": fn}


def segment_score(primary, secondary, *, radius=10, threshold=0.5, mode="legacy_exact"):
    """Length-weighted connected-segment matching (8-connectivity, square buffer).

    legacy_exact retains CTS.py's initial unbuffered secondary intersection.
    buffered replaces this with the buffer of each secondary component; it is
    a separate diagnostic definition, not a silent correction to paper results.
    """
    if radius < 0 or not isinstance(radius, int) or not 0 <= threshold <= 1 or mode not in ("legacy_exact", "buffered"):
        raise ValueError("Invalid CTS protocol")
    p_labels, s_labels = label(primary, connectivity=2), label(secondary, connectivity=2)
    p_lengths = np.bincount(p_labels.ravel())
    total = int(primary.sum())
    if total == 0 or not secondary.any():
        return 0.0
    size = 2*radius+1
    p_buffer = maximum_filter(primary, size=size, mode="constant", cval=0)
    s_buffer = maximum_filter(secondary, size=size, mode="constant", cval=0)
    secondary_components = [s_labels == i for i in range(1, int(s_labels.max())+1)]
    secondary_buffers = [maximum_filter(s, size=size, mode="constant", cval=0)
                         for s in secondary_components] if mode == "buffered" else secondary_components
    matched_length = 0
    for component_id in range(1, len(p_lengths)):
        segment = p_labels == component_id
        false_positive = int((segment & ~s_buffer).sum())
        overlaps = []
        for index, (other, other_buffer) in enumerate(zip(secondary_components, secondary_buffers)):
            tp = int((segment & s_buffer & other_buffer).sum())
            fn = int((other & ~p_buffer).sum())
            correctness = ratio(tp, tp+false_positive, empty_value=0)
            completeness = ratio(tp, tp+fn, empty_value=0)
            if correctness > threshold or completeness > threshold:
                overlaps.append(index)
        if not overlaps:
            continue
        merged = np.logical_or.reduce([secondary_components[i] for i in overlaps])
        merged_buffer = maximum_filter(merged, size=size, mode="constant", cval=0)
        correctness = float((segment & merged_buffer).sum())/int(p_lengths[component_id])
        if correctness > threshold:
            matched_length += int(p_lengths[component_id])
    return matched_length/total


def compute_cts(prediction, target, *, radius=10, threshold=0.5, mode="legacy_exact", already_skeleton=False):
    prediction, target = masks(prediction, target)
    if not already_skeleton:
        prediction = skeletonize(prediction, method="zhang")
        target = skeletonize(target, method="zhang")
    pcs = segment_score(prediction, target, radius=radius, threshold=threshold, mode=mode)
    rcs = segment_score(target, prediction, radius=radius, threshold=threshold, mode=mode)
    return {"pcs": pcs, "rcs": rcs, "cts": ratio(2*pcs*rcs, pcs+rcs, empty_value=0)}


def score_image(prediction, target, *, deltas=(0, 2, 4, 8), cts_radius=10,
                cts_threshold=0.5, cts_mode="legacy_exact", empty_iou=1.0, empty_cliou=1.0,
                cliou_skeleton="thin", cliou_empty_policy="include", empty_cts=0.0):
    prediction, target = masks(prediction, target)
    intersection, union = int((prediction & target).sum()), int((prediction | target).sum())
    p_skel, g_skel = skeletonize(prediction, method="zhang"), skeletonize(target, method="zhang")
    if cliou_skeleton not in ("thin", "zhang") or cliou_empty_policy not in ("include", "exclude"):
        raise ValueError("Unknown legacy CL-IoU protocol")
    cl_pred, cl_gt = (thin(prediction), thin(target)) if cliou_skeleton == "thin" else (p_skel, g_skel)
    result = {"iou": ratio(intersection, union, empty_value=empty_iou),
              "iou_intersection": intersection, "iou_union": union,
              "pred_foreground": int(prediction.sum()), "gt_foreground": int(target.sum())}
    for delta in deltas:
        counts = centerline_counts(cl_pred, cl_gt, delta)
        key = f"cliou_{delta}"
        result[key] = ratio(counts["tp"], sum(counts.values()), empty_value=empty_cliou)
        result[f"{key}_included"] = cliou_empty_policy == "include" or sum(counts.values()) > 0
        result.update({f"{key}_{name}": value for name, value in counts.items()})
    result.update(compute_cts(p_skel, g_skel, radius=cts_radius, threshold=cts_threshold,
                              mode=cts_mode, already_skeleton=True))
    if not p_skel.any() and not g_skel.any():
        result.update(pcs=float(empty_cts), rcs=float(empty_cts), cts=float(empty_cts))
    return result


class MetricAccumulator:
    """Image macro means and global micro counts; no shared train/val/test state."""
    def __init__(self, *, deltas=(0, 2, 4, 8), **options):
        self.deltas, self.options = tuple(deltas), options
        if len(set(self.deltas)) != len(self.deltas):
            raise ValueError("Repeated CL-IoU tolerance")
        self.rows = []
        self.ids = set()

    def update(self, sample_id, prediction, target):
        if sample_id in self.ids:
            raise ValueError(f"Sample evaluated twice: {sample_id}")
        scores = score_image(prediction, target, deltas=self.deltas, **self.options)
        self.ids.add(sample_id)
        self.rows.append({"sample_id": sample_id, **scores})
        return scores

    def summary(self):
        if not self.rows:
            raise ValueError("Cannot report scores for zero samples")
        names = ("iou", "pcs", "rcs", "cts", *(f"cliou_{d}" for d in self.deltas))
        macro, macro_counts = {}, {}
        for name in names:
            included = [r[name] for r in self.rows if r.get(f"{name}_included", True)]
            macro_counts[name] = len(included)
            macro[name] = math.fsum(included)/len(included) if included else 0.0
        micro = {"iou": ratio(sum(r["iou_intersection"] for r in self.rows),
                               sum(r["iou_union"] for r in self.rows),
                               empty_value=self.options.get("empty_iou", 1.0))}
        for delta in self.deltas:
            key = f"cliou_{delta}"
            tp = sum(r[f"{key}_tp"] for r in self.rows)
            denominator = sum(r[f"{key}_{name}"] for r in self.rows for name in ("tp", "fp", "fn"))
            micro[key] = ratio(tp, denominator, empty_value=self.options.get("empty_cliou", 1.0))
        return {"count": len(self.rows), "macro": macro, "macro_counts": macro_counts, "micro": micro,
                "empty_gt_count": sum(r["gt_foreground"] == 0 for r in self.rows),
                "false_positive_on_empty_gt_count": sum(r["gt_foreground"] == 0 and r["pred_foreground"] > 0 for r in self.rows)}
