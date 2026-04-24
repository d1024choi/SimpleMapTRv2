import numpy as np
import torch

from utils.metrics import PolylinemAPMetric
from utils.map_mean_ap_standalone import PolylineMapAPStandalone


def build_dummy_batch(
    batch_size: int,
    num_queries: int,
    num_points: int,
    class_names,
    has_background_class: bool = True,
):
    num_fg = len(class_names)
    num_classes = num_fg + 1 if has_background_class else num_fg

    # Random predictions
    pred_polylines = torch.rand(batch_size, num_queries, num_points, 2) * 10.0
    pred_scores = torch.randn(batch_size, num_queries, num_classes)

    # Simple GT: for each batch item, create a frame with a few polylines
    gt_polylines = []
    for b in range(batch_size):
        N = num_fg  # one GT per class
        polys = torch.rand(N, num_points, 2) * 10.0
        labels = torch.arange(num_fg, dtype=torch.long)
        frame_idx = 0
        gt_item = {frame_idx: {"polylines": polys, "labels": labels}}
        gt_polylines.append(gt_item)

    return pred_polylines, pred_scores, gt_polylines


def main():
    class_names = ["drivable", "divider", "ped_crossing"]
    has_bg = True

    pred_pl, pred_sc, gt_pl = build_dummy_batch(
        batch_size=2,
        num_queries=64,
        num_points=50,
        class_names=class_names,
        has_background_class=has_bg,
    )

    # Original implementation
    metric_orig = PolylinemAPMetric(
        class_names=class_names,
        chamfer_thresholds=[0.5, 1.0, 1.5],
        num_sample_pts=100,
        metric="chamfer",
        has_background_class=has_bg,
    )
    metric_orig.update(pred_pl, pred_sc, gt_pl)
    res_orig = metric_orig.compute()

    # Standalone implementation
    metric_sa = PolylineMapAPStandalone(
        class_names=class_names,
        chamfer_thresholds=[0.5, 1.0, 1.5],
        num_sample_pts=100,
        metric="chamfer",
        has_background_class=has_bg,
    )
    metric_sa.update(pred_pl, pred_sc, gt_pl)
    res_sa = metric_sa.compute()

    keys = sorted(set(res_orig.keys()) | set(res_sa.keys()))
    all_close = True
    for k in keys:
        v1 = res_orig.get(k, None)
        v2 = res_sa.get(k, None)
        if v1 is None or v2 is None:
            print(f"{k}: missing in one result (orig={v1}, sa={v2})")
            all_close = False
            continue
        if not np.allclose(float(v1), float(v2), atol=1e-6):
            print(f"{k}: mismatch orig={v1}, sa={v2}")
            all_close = False

    if all_close:
        print("Validation PASSED: PolylineMapAPStandalone matches PolylinemAPMetric.")
        print(res_sa)
    else:
        print("Validation FAILED: differences detected between implementations.")
        print("orig:", res_orig)
        print("sa  :", res_sa)


if __name__ == "__main__":
    main()

