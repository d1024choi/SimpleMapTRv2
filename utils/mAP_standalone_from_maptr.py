import numpy as np
from multiprocessing import Pool
from functools import partial
from shapely.geometry import LineString
import torch
from scipy.spatial import distance as scipy_dist


def _average_precision(recalls, precisions, mode: str = "area"):
    """Standalone copy of MapTR average_precision (area / 11points)."""
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == "area":
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == "11points":
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError('Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def _get_cls_results(
    gen_results,
    annotations,
    num_sample: int = 100,
    num_pred_pts_per_instance: int = 30,
    eval_use_same_gt_sample_num_flag: bool = False,
    class_id: int = 0,
    fix_interval: bool = False,
):
    """Standalone copy of get_cls_results from MapTR mean_ap.py."""
    cls_gens, cls_scores = [], []
    for res in gen_results["vectors"]:
        if res["type"] == class_id:
            if len(res["pts"]) < 2:
                continue
            if not eval_use_same_gt_sample_num_flag:
                sampled_points = np.array(res["pts"])
            else:
                line = res["pts"]
                line = LineString(line)
                if fix_interval:
                    distances = list(np.arange(1.0, line.length, 1.0))
                    distances = [0.0] + distances + [line.length]
                    sampled_points = (
                        np.array([list(line.interpolate(distance).coords) for distance in distances])
                        .reshape(-1, 2)
                    )
                else:
                    distances = np.linspace(0.0, line.length, num_sample)
                    sampled_points = (
                        np.array([list(line.interpolate(distance).coords) for distance in distances])
                        .reshape(-1, 2)
                    )
            cls_gens.append(sampled_points)
            cls_scores.append(res["confidence_level"])
    num_res = len(cls_gens)
    if num_res > 0:
        cls_gens = np.stack(cls_gens).reshape(num_res, -1)
        cls_scores = np.array(cls_scores)[:, np.newaxis]
        cls_gens = np.concatenate([cls_gens, cls_scores], axis=-1)
    else:
        if not eval_use_same_gt_sample_num_flag:
            cls_gens = np.zeros((0, num_pred_pts_per_instance * 2 + 1))
        else:
            cls_gens = np.zeros((0, num_sample * 2 + 1))

    cls_gts = []
    for ann in annotations["vectors"]:
        if ann["type"] == class_id:
            line = ann["pts"]
            line = LineString(line)
            distances = np.linspace(0.0, line.length, num_sample)
            sampled_points = (
                np.array([list(line.interpolate(distance).coords) for distance in distances])
                .reshape(-1, 2)
            )
            cls_gts.append(sampled_points)
    num_gts = len(cls_gts)
    if num_gts > 0:
        cls_gts = np.stack(cls_gts).reshape(num_gts, -1)
    else:
        cls_gts = np.zeros((0, num_sample * 2))
    return cls_gens, cls_gts


class StandaloneMapMeanAP:
    """Standalone evaluator replicating MapTR mean_ap.eval_map behavior.

    This class does not depend on MMCV or MMDetection; it only uses NumPy,
    multiprocessing, and Shapely. It expects `gen_results` and `annotations`
    in the same format as MapTR's map vector evaluation.
    """

    def __init__(self, nproc: int = 24):
        self.nproc = nproc

    def format_res_gt_by_classes(
        self,
        gen_results,
        annotations,
        cls_names,
        num_pred_pts_per_instance: int = 30,
        eval_use_same_gt_sample_num_flag: bool = False,
        num_fixed_sample_pts: int = 100,
        fix_interval: bool = False,
    ):
        assert cls_names is not None
        assert len(gen_results) == len(annotations)

        pool = Pool(self.nproc)
        cls_gens, cls_gts = {}, {}

        for i, clsname in enumerate(cls_names):
            gengts = pool.starmap(
                partial(
                    _get_cls_results,
                    num_sample=num_fixed_sample_pts,
                    num_pred_pts_per_instance=num_pred_pts_per_instance,
                    eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
                    class_id=i,
                    fix_interval=fix_interval,
                ),
                zip(gen_results, annotations),
            )
            gens, gts = tuple(zip(*gengts))
            cls_gens[clsname] = gens
            cls_gts[clsname] = gts

        pool.close()
        return cls_gens, cls_gts

    def eval_map(
        self,
        gen_results,
        annotations,
        cls_gens,
        cls_gts,
        threshold: float = 0.5,
        cls_names=None,
        metric=None,
        num_pred_pts_per_instance: int = 30,
    ):
        """Compute mean AP and per-class AP, replicating MapTR eval_map."""
        from projects.mmdet3d_plugin.datasets.map_utils.tpfp import (  # type: ignore  # noqa: E501
            custom_tpfp_gen,
        )

        pool = Pool(self.nproc)
        eval_results = []

        for i, clsname in enumerate(cls_names):
            cls_gen = cls_gens[clsname]
            cls_gt = cls_gts[clsname]

            tpfp_fn = partial(custom_tpfp_gen, threshold=threshold, metric=metric)
            tpfp = pool.starmap(tpfp_fn, zip(cls_gen, cls_gt))
            tp, fp = tuple(zip(*tpfp))

            num_gts = 0
            for bbox in cls_gt:
                num_gts += bbox.shape[0]

            cls_gen_flat = np.vstack(cls_gen)
            num_dets = cls_gen_flat.shape[0]
            sort_inds = np.argsort(-cls_gen_flat[:, -1])
            tp = np.hstack(tp)[sort_inds]
            fp = np.hstack(fp)[sort_inds]

            tp = np.cumsum(tp, axis=0)
            fp = np.cumsum(fp, axis=0)
            eps = np.finfo(np.float32).eps
            recalls = tp / np.maximum(num_gts, eps)
            precisions = tp / np.maximum((tp + fp), eps)

            ap = _average_precision(recalls, precisions, mode="area")
            eval_results.append(
                {
                    "num_gts": num_gts,
                    "num_dets": num_dets,
                    "recall": recalls,
                    "precision": precisions,
                    "ap": ap,
                }
            )

        pool.close()

        aps = []
        for cls_result in eval_results:
            if cls_result["num_gts"] > 0:
                aps.append(cls_result["ap"])
        mean_ap = np.array(aps).mean().item() if len(aps) else 0.0
        return mean_ap, eval_results


class PolylineMapAPStandalone:
    """Standalone clone of PolylinemAPMetric from utils.metrics.

    This class exposes the same interface:
        - __init__(class_names, chamfer_thresholds, num_sample_pts, metric, has_background_class)
        - update(pred_polylines, pred_scores, gt_polylines)
        - compute() -> dict with 'mAP' and per-class 'AP/<name>'
        - reset()

    It is implemented here to allow unit-testing against the original implementation
    without importing the whole metrics module, and to decouple downstream code from
    other metric utilities.
    """

    def __init__(
        self,
        class_names,
        chamfer_thresholds=(0.5, 1.0, 1.5),
        num_sample_pts: int = 100,
        metric: str = "chamfer",
        has_background_class: bool = False,
    ):
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)  # foreground classes only
        self.chamfer_thresholds = list(chamfer_thresholds)
        self.num_sample_pts = num_sample_pts
        self.metric = metric
        self.has_background_class = has_background_class

        # Accumulated state, same layout as PolylinemAPMetric
        self._pred_scores_by_cls = {c: [] for c in range(self.num_classes)}
        self._tp_by_cls_thr = {c: {thr: [] for thr in chamfer_thresholds} for c in range(self.num_classes)}
        self._fp_by_cls_thr = {c: {thr: [] for thr in chamfer_thresholds} for c in range(self.num_classes)}
        self._num_gts_by_cls = {c: 0 for c in range(self.num_classes)}

    # ---- helpers cloned from utils.metrics.PolylinemAPMetric ----

    def _ensure_pts_xy(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2:
            return np.zeros((0, 2), dtype=np.float64)
        if pts.shape[0] == 2 and pts.shape[1] != 2:
            pts = pts.T
        if pts.shape[1] != 2:
            return np.zeros((0, 2), dtype=np.float64)
        return pts

    def _resample_polyline(self, pts: np.ndarray, n: int) -> np.ndarray:
        pts = self._ensure_pts_xy(pts)
        if len(pts) < 2:
            return np.zeros((n, 2), dtype=np.float64)
        line = LineString(pts)
        if line.length < 1e-6:
            return np.tile(pts[0:1], (n, 1))
        dists = np.linspace(0, line.length, n, endpoint=(n > 1))
        return np.array([list(line.interpolate(d).coords)[0] for d in dists], dtype=np.float64)

    def _polyline_score_matrix(self, pred_lines: np.ndarray, gt_lines: np.ndarray) -> np.ndarray:
        def _chamfer_score(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
            if pred_pts.shape[0] < 2 or gt_pts.shape[0] < 2:
                return -100.0
            d = scipy_dist.cdist(pred_pts, gt_pts, "euclidean")
            d_ab = d.min(axis=1).mean()
            d_ba = d.min(axis=0).mean()
            chamfer = (d_ab + d_ba) / 2.0
            return -float(chamfer)

        n_pred, n_gt = len(pred_lines), len(gt_lines)
        if n_pred == 0 or n_gt == 0:
            return np.zeros((n_pred, n_gt), dtype=np.float64)
        mat = np.full((n_pred, n_gt), -100.0, dtype=np.float64)
        for i in range(n_pred):
            for j in range(n_gt):
                if self.metric == "chamfer":
                    mat[i, j] = _chamfer_score(pred_lines[i], gt_lines[j])
                else:
                    raise SystemExit("PolylineMapAPStandalone: metric == iou is not supported yet.")
        return mat

    def _compute_tp_fp(
        self,
        pred_lines: np.ndarray,
        pred_scores: np.ndarray,
        gt_lines: np.ndarray,
        threshold: float,
    ):
        n_pred, n_gt = len(pred_lines), len(gt_lines)
        tp = np.zeros(n_pred, dtype=np.float32)
        fp = np.zeros(n_pred, dtype=np.float32)
        if n_gt == 0:
            fp[:] = 1
            return tp, fp
        if n_pred == 0:
            return tp, fp
        thr = -threshold if self.metric == "chamfer" else threshold

        score_mat = self._polyline_score_matrix(pred_lines, gt_lines)
        mat_max = score_mat.max(axis=1)
        mat_argmax = score_mat.argmax(axis=1)
        sort_inds = np.argsort(-pred_scores)  # descending
        gt_covered = np.zeros(n_gt, dtype=bool)

        for i in sort_inds:
            if mat_max[i] >= thr:
                m = mat_argmax[i]
                if not gt_covered[m]:
                    gt_covered[m] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        return tp, fp

    def _average_precision(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        recalls = np.asarray(recalls, dtype=np.float64)
        precisions = np.asarray(precisions, dtype=np.float64)
        mrec = np.concatenate([[0], recalls, [1]])
        mpre = np.concatenate([[0], precisions, [0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    # ---- public API matching PolylinemAPMetric ----

    def _extract_gt_by_class(self, gt_polylines_item):
        out = {c: [] for c in range(self.num_classes)}
        if not isinstance(gt_polylines_item, dict) or len(gt_polylines_item) == 0:
            return out
        _, frame_data = max(gt_polylines_item.items(), key=lambda x: x[0])
        if isinstance(frame_data, dict) and "polylines" in frame_data and "labels" in frame_data:
            polylines = frame_data["polylines"]
            labels = frame_data["labels"]
            if not hasattr(polylines, "__len__") or len(polylines) == 0:
                return out
            if not hasattr(labels, "__len__") or len(labels) == 0:
                return out
            pl_np = polylines.cpu().numpy() if torch.is_tensor(polylines) else np.asarray(polylines)
            lb_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
            for idx in range(len(lb_np)):
                cls_id = int(lb_np[idx])
                if cls_id < 0 or cls_id >= self.num_classes:
                    continue
                pts = self._ensure_pts_xy(pl_np[idx])
                if len(pts) < 2:
                    continue
                out[cls_id].append(self._resample_polyline(pts, self.num_sample_pts))
            return out
        return out

    def _extract_pred_by_class(self, pred_pl: np.ndarray, pred_sc: np.ndarray):
        out = {c: ([], []) for c in range(self.num_classes)}
        logits = torch.from_numpy(pred_sc.astype(np.float32))
        if self.has_background_class:
            probs = logits.softmax(dim=-1).to("cpu").numpy()
            fg_probs = probs[:, : self.num_classes]
        else:
            fg_probs = logits[:, : self.num_classes].sigmoid().to("cpu").numpy()

        for q in range(pred_pl.shape[0]):
            cls_id = int(np.argmax(fg_probs[q]))
            score = float(fg_probs[q, cls_id])
            if cls_id < 0 or cls_id >= self.num_classes:
                continue
            pts = self._ensure_pts_xy(pred_pl[q])
            if len(pts) < 2:
                continue
            resampled = self._resample_polyline(pts, self.num_sample_pts)
            out[cls_id][0].append(resampled)
            out[cls_id][1].append(score)
        return out

    def update(self, pred_polylines, pred_scores, gt_polylines):
        pred_pl = (
            pred_polylines.detach().cpu().numpy()
            if torch.is_tensor(pred_polylines)
            else np.asarray(pred_polylines)
        )
        pred_sc = (
            pred_scores.detach().cpu().numpy()
            if torch.is_tensor(pred_scores)
            else np.asarray(pred_scores)
        )
        B = pred_pl.shape[0]
        for b in range(B):
            gts_by_cls = self._extract_gt_by_class(gt_polylines[b])
            preds_by_cls = self._extract_pred_by_class(pred_pl[b], pred_sc[b])
            for cls_id in range(self.num_classes):
                gt_lines = gts_by_cls[cls_id]
                pred_lines, pred_scores_list = preds_by_cls[cls_id]
                self._num_gts_by_cls[cls_id] += len(gt_lines)
                if len(pred_lines) == 0 and len(gt_lines) == 0:
                    continue
                if len(pred_lines) == 0:
                    continue
                pred_lines = np.stack(pred_lines, axis=0)
                pred_scores_arr = np.array(pred_scores_list, dtype=np.float32)
                gt_arr = (
                    np.stack(gt_lines, axis=0)
                    if len(gt_lines) > 0
                    else np.zeros((0, self.num_sample_pts, 2), dtype=np.float64)
                )
                self._pred_scores_by_cls[cls_id].append(pred_scores_arr)
                for thr in self.chamfer_thresholds:
                    tp, fp = self._compute_tp_fp(pred_lines, pred_scores_arr, gt_arr, thr)
                    self._tp_by_cls_thr[cls_id][thr].append(tp)
                    self._fp_by_cls_thr[cls_id][thr].append(fp)

    def compute(self):
        result = {}
        aps_per_cls = []
        for cls_id in range(self.num_classes):
            scores_list = self._pred_scores_by_cls[cls_id]
            num_gts = self._num_gts_by_cls[cls_id]
            if num_gts == 0 or len(scores_list) == 0:
                result[f"AP/{self.class_names[cls_id]}"] = 0.0
                continue
            all_scores = np.concatenate(scores_list)
            sort_inds = np.argsort(-all_scores)
            ap_per_thr = []
            for thr in self.chamfer_thresholds:
                tp_list = self._tp_by_cls_thr[cls_id][thr]
                fp_list = self._fp_by_cls_thr[cls_id][thr]
                if len(tp_list) == 0:
                    ap_per_thr.append(0.0)
                    continue
                all_tp = np.concatenate(tp_list)
                all_fp = np.concatenate(fp_list)
                tp = np.cumsum(all_tp[sort_inds])
                fp = np.cumsum(all_fp[sort_inds])
                eps = np.finfo(np.float32).eps
                recalls = tp / max(num_gts, eps)
                precisions = tp / np.maximum(tp + fp, eps)
                ap_per_thr.append(self._average_precision(recalls, precisions))
            ap = float(np.mean(ap_per_thr))
            result[f"AP/{self.class_names[cls_id]}"] = ap
            aps_per_cls.append(ap)
        result["mAP"] = float(np.mean(aps_per_cls)) if aps_per_cls else 0.0
        return result

    def reset(self):
        for c in range(self.num_classes):
            self._pred_scores_by_cls[c] = []
            for thr in self.chamfer_thresholds:
                self._tp_by_cls_thr[c][thr] = []
                self._fp_by_cls_thr[c][thr] = []
            self._num_gts_by_cls[c] = 0

