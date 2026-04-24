import torch
import cv2
import numpy as np
import re
import sys
from torchmetrics import Metric
from typing import List, Optional, Dict, Tuple, Union

try:
    from scipy.spatial import distance as scipy_dist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from shapely.geometry import LineString
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

# try:
#     from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#     import nltk
#     # Download required NLTK data if not already present
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt', quiet=True)
#     try:
#         nltk.data.find('tokenizers/averaged_perceptron_tagger')
#     except LookupError:
#         nltk.download('averaged_perceptron_tagger', quiet=True)
#     try:
#         nltk.data.find('corpora/wordnet')
#     except LookupError:
#         nltk.download('wordnet', quiet=True)
#     NLTK_AVAILABLE = True
# except ImportError:
#     NLTK_AVAILABLE = False
#     print("Warning: NLTK not available. Text quality metrics will use simple tokenization.")


class BaseIoUMetric(torch.nn.Module):
    """
    Computes intersection over union at given thresholds
    """
    def __init__(self, thresholds=[0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]):
        super().__init__()

        self.thresholds = torch.from_numpy(np.array(thresholds))
        self.tp = torch.zeros_like(self.thresholds)
        self.fp = torch.zeros_like(self.thresholds)
        self.fn = torch.zeros_like(self.thresholds)

    def update(self, pred, label, isLogit=True):

        if (isLogit): pred = pred.detach().to('cpu').sigmoid().reshape(-1)
        else: pred = pred.detach().to('cpu').reshape(-1)
        label = label.detach().to('cpu').bool().reshape(-1)

        pred = pred[:, None] >= self.thresholds[None]
        label = label[:, None]

        self.tp += (pred & label).sum(0)
        self.fp += (pred & ~label).sum(0)
        self.fn += (~pred & label).sum(0)

    def compute(self):
        thresholds = self.thresholds.squeeze(0)
        ious = self.tp / (self.tp + self.fp + self.fn + 1e-7)

        output = {}
        for t, i in zip(thresholds, ious):
            output.update({f'@{t.item():.2f}': i.item()})

        return {f'@{t.item():.2f}': i.item() for t, i in zip(thresholds, ious)}


class IoUMetric(BaseIoUMetric):
    def __init__(self, label_indices: List[List[int]],
                 min_visibility: Optional[int] = None,
                 target_class: Optional[str] = None): # update 231006
        """
        label_indices:
            transforms labels (c, h, w) to (len(labels), h, w)
            see config/experiment/* for examples

        min_visibility:
            passing "None" will ignore the visibility mask
            otherwise uses visibility values to ignore certain labels
            visibility mask is in order of "increasingly visible" {1, 2, 3, 4, 255 (default)}
            see https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#visibility
        """
        super().__init__()

        self.label_indices = label_indices
        self.min_visibility = min_visibility
        self.target_class = target_class

    def update(self, pred, batch):

        label = batch['bev']                                                                # b n h w
        label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
        label = torch.cat(label, 1)                                                         # b c h w

        # update 231006
        visibility = None
        if (self.target_class == 'vehicle'): visibility = batch['visibility'][:, [0]]
        elif (self.target_class == 'pedestrian'): visibility = batch['visibility'][:, [1]]

        if self.min_visibility is not None:
            mask = visibility >= self.min_visibility
            mask = mask.expand_as(pred)                                            # b c h w

            pred = pred[mask]                                                               # m
            label = label[mask]                                                             # m

        return super().update(pred, label)


class TextQualityMetric:
    """
    Computes text quality metrics (BLEU, ROUGE-L) for generated prompts.
    """
    def __init__(self):
        self.smoothing = SmoothingFunction().method1 if NLTK_AVAILABLE else None
        
    def tokenize(self, text):
        """Tokenize text into words."""
        if NLTK_AVAILABLE:
            return nltk.word_tokenize(text.lower())
        else:
            # Simple tokenization fallback
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            return text.split()
    
    def compute_bleu(self, reference, candidate):
        """Compute BLEU-4 score between reference and candidate."""
        if not NLTK_AVAILABLE:
            # Simple fallback: exact match check
            ref_tokens = self.tokenize(reference)
            cand_tokens = self.tokenize(candidate)
            if len(ref_tokens) == 0 or len(cand_tokens) == 0:
                return 0.0
            matches = sum(1 for t in cand_tokens if t in ref_tokens)
            return matches / max(len(cand_tokens), 1)
        
        ref_tokens = [self.tokenize(reference)]
        cand_tokens = self.tokenize(candidate)
        
        if len(ref_tokens[0]) == 0 or len(cand_tokens) == 0:
            return 0.0
            
        return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=self.smoothing)
    
    def compute_rouge_l(self, reference, candidate):
        """Compute ROUGE-L score (longest common subsequence)."""
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        
        if len(ref_tokens) == 0 or len(cand_tokens) == 0:
            return 0.0
        
        # Compute LCS
        lcs_length = self._lcs_length(ref_tokens, cand_tokens)
        
        # ROUGE-L = LCS length / (length of reference + length of candidate)
        precision = lcs_length / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
        recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score
    
    def _lcs_length(self, seq1, seq2):
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def compute_metrics(self, references, candidates):
        """
        Compute average BLEU and ROUGE-L scores.
        
        Args:
            references: List of reference text strings
            candidates: List of candidate/generated text strings
            
        Returns:
            dict with 'bleu' and 'rouge_l' scores
        """
        if len(references) != len(candidates):
            raise ValueError(f"References and candidates must have same length. Got {len(references)} and {len(candidates)}")
        
        if len(references) == 0:
            return {'bleu': 0.0, 'rouge_l': 0.0}
        
        bleu_scores = []
        rouge_l_scores = []
        
        for ref, cand in zip(references, candidates):
            # Skip empty references or candidates
            if not ref or not cand:
                continue
                
            bleu = self.compute_bleu(ref, cand)
            rouge_l = self.compute_rouge_l(ref, cand)
            
            bleu_scores.append(bleu)
            rouge_l_scores.append(rouge_l)
        
        if len(bleu_scores) == 0:
            return {'bleu': 0.0, 'rouge_l': 0.0}
        
        avg_bleu = np.mean(bleu_scores)
        avg_rouge_l = np.mean(rouge_l_scores)
        
        return {
            'bleu': float(avg_bleu),
            'rouge_l': float(avg_rouge_l),
            'bleu_std': float(np.std(bleu_scores)),
            'rouge_l_std': float(np.std(rouge_l_scores))
        }


class PolylinemAPMetric:
    """
    Computes mAP and per-class AP for 3D lane / polyline prediction.
    Uses Chamfer distance-based matching (Mask2Map-style) for TP/FP assignment.

    Following the Mask2Map evaluation protocol, ALL predictions are included in the
    AP computation without any score-based pre-filtering. AP is determined solely by
    the ranking of predictions (by confidence) and the Chamfer distance thresholds
    for TP/FP matching.

    Expects:
        - pred_polylines: (B, num_queries, num_points, 2) tensor in BEV metre coordinates
        - pred_scores: (B, num_queries, num_classes) raw logits. When has_background_class=True,
          num_classes = num_fg + 1 (last is background); probabilities are obtained via softmax.
          When has_background_class=False, logits are treated with sigmoid (foreground-only).
        - gt_polylines: List of length B. Two formats are accepted (auto-detected):

            loader_typeA — {frame_idx: {'polylines': Tensor(N,P,2), 'labels': Tensor(N,),
                                        'bboxes': Tensor(N,4), 'polylines_shift': Tensor(N,R,P,2)}}
            loader_typeB — {frame_idx: [{'gt_label': int, 'p_coords': ndarray, ...}]}
    """

    def __init__(
        self,
        class_names: List[str],
        chamfer_thresholds: List[float] = (0.5, 1.0, 1.5),
        num_sample_pts: int = 100,
        metric: str = 'chamfer',
        has_background_class: bool = False,
    ):
        """
        Args:
            class_names: List of class names (foreground only), e.g. ['drivable', 'divider', 'ped_crossing']
            chamfer_thresholds: Chamfer distance thresholds in meters for matching (pred matches GT if dist <= thr)
            num_sample_pts: Number of points to resample polylines for fair Chamfer comparison
            metric: 'chamfer' (default) or 'iou' (requires shapely)
            has_background_class: If True, pred_scores are multi-class logits (last = background).
                Probabilities are computed with softmax; background is excluded from assignment.
                GT labels are expected to be foreground class indices (0 .. num_fg_classes-1).
        """
        assert SCIPY_AVAILABLE, "PolylinemAPMetric requires scipy"
        if metric == 'iou':
            assert SHAPELY_AVAILABLE, "IoU metric requires shapely"

        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)  # foreground classes only
        self.chamfer_thresholds = list(chamfer_thresholds)
        self.num_sample_pts = num_sample_pts
        self.metric = metric
        self.has_background_class = has_background_class

        # Accumulate per-class, per-threshold: (scores, tp, fp) for each threshold
        self._pred_scores_by_cls: Dict[int, List[np.ndarray]] = {c: [] for c in range(self.num_classes)}
        self._tp_by_cls_thr: Dict[int, Dict[float, List[np.ndarray]]] = {
            c: {thr: [] for thr in chamfer_thresholds} for c in range(self.num_classes)
        }
        self._fp_by_cls_thr: Dict[int, Dict[float, List[np.ndarray]]] = {
            c: {thr: [] for thr in chamfer_thresholds} for c in range(self.num_classes)
        }
        self._num_gts_by_cls: Dict[int, int] = {c: 0 for c in range(self.num_classes)}

    def _ensure_pts_xy(self, pts: np.ndarray) -> np.ndarray:
        """Ensure shape (N, 2) for (x, y)."""
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2:
            return np.zeros((0, 2), dtype=np.float64)
        if pts.shape[0] == 2 and pts.shape[1] != 2:
            pts = pts.T  # (2, M) -> (M, 2)
        if pts.shape[1] != 2:
            return np.zeros((0, 2), dtype=np.float64)
        return pts

    def _resample_polyline(self, pts: np.ndarray, n: int) -> np.ndarray:
        """Resample polyline to n points uniformly along length. pts: (M, 2)."""
        pts = self._ensure_pts_xy(pts)
        if len(pts) < 2:
            return np.zeros((n, 2), dtype=np.float64)
        if SHAPELY_AVAILABLE:
            line = LineString(pts)
            if line.length < 1e-6:
                return np.tile(pts[0:1], (n, 1))
            dists = np.linspace(0, line.length, n, endpoint=(n > 1))
            return np.array([list(line.interpolate(d).coords)[0] for d in dists], dtype=np.float64)
        # Fallback: linear interpolation along cumulative arc length
        d = np.zeros(len(pts) + 1)
        d[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        if d[-1] < 1e-6:
            return np.tile(pts[0:1], (n, 1))
        t = np.linspace(0, d[-1], n, endpoint=(n > 1))
        return np.interp(t, d[:-1], pts.T).T

    def _polyline_score_matrix(
        self, pred_lines: np.ndarray, gt_lines: np.ndarray, linewidth: float = 2.0
    ) -> np.ndarray:
        """Score matrix (num_preds, num_gts). For chamfer: higher = better."""

        def _chamfer_score(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
            """Symmetric Chamfer distance; return negative (higher = better match)."""
            if pred_pts.shape[0] < 2 or gt_pts.shape[0] < 2:
                return -100.0
            d = scipy_dist.cdist(pred_pts, gt_pts, 'euclidean') # (pred_pts.shape[0], gt_pts.shape[0])
            d_ab = d.min(axis=1).mean()
            d_ba = d.min(axis=0).mean()
            chamfer = (d_ab + d_ba) / 2
            return -float(chamfer)

        n_pred, n_gt = len(pred_lines), len(gt_lines)
        if n_pred == 0 or n_gt == 0:
            return np.zeros((n_pred, n_gt), dtype=np.float64)
        mat = np.full((n_pred, n_gt), -100.0, dtype=np.float64)
        for i in range(n_pred):
            for j in range(n_gt):
                if self.metric == 'chamfer':
                    # The highter, the better match
                    mat[i, j] = _chamfer_score(pred_lines[i], gt_lines[j])
                else:
                    sys.exit('PolylinemAPMetric: metric == iou is not supported yet.')
        return mat

    def _compute_tp_fp(
        self,
        pred_lines: np.ndarray,
        pred_scores: np.ndarray,
        gt_lines: np.ndarray,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        TP/FP arrays of length num_preds. Uses greedy matching by score.
        TP condition : 1) class match, 2) dist(pred, gt) <= threshold
        FP condition : 1) class mismatch, 2) dist(pred, gt) <= threshold, 3) gt is already covered
        '''
        n_pred, n_gt = len(pred_lines), len(gt_lines)
        tp = np.zeros(n_pred, dtype=np.float32)
        fp = np.zeros(n_pred, dtype=np.float32)
        if n_gt == 0:
            fp[:] = 1
            return tp, fp
        if n_pred == 0:
            return tp, fp
        thr = -threshold if self.metric == 'chamfer' else threshold
        
        # score_mat: (num_preds, num_gts), the higher, the better match
        score_mat = self._polyline_score_matrix(pred_lines, gt_lines) 
        mat_max = score_mat.max(axis=1)
        mat_argmax = score_mat.argmax(axis=1) # saying, which gt is assigned to the current pred
        sort_inds = np.argsort(-pred_scores) # descending, high score front
        gt_covered = np.zeros(n_gt, dtype=bool)

        # multiple predictions can be assigned to the same gt,
        # therefore, we need to check higher score predictions first
        for i in sort_inds:
            if mat_max[i] >= thr: # dist(pred, gt) <= threshold
                m = mat_argmax[i] # which gt is assigned to the current pred
                if not gt_covered[m]:
                    gt_covered[m] = True
                    tp[i] = 1
                else:
                    # Even though the distance is less than threshold,
                    # if the gt is already covered, it is a false positive
                    fp[i] = 1
            else:
                fp[i] = 1
        return tp, fp

    def _average_precision(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        """Area under PR curve."""
        recalls = np.asarray(recalls, dtype=np.float64)
        precisions = np.asarray(precisions, dtype=np.float64)
        mrec = np.concatenate([[0], recalls, [1]])
        mpre = np.concatenate([[0], precisions, [0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    def _extract_gt_by_class(self, gt_polylines_item) -> Dict[int, List[np.ndarray]]:
        """Extract GT polylines per class from one batch item.

        Supports two loader formats (auto-detected):

        **loader_typeA** (NuScenes, tensor-based):
            ``{frame_idx: {'polylines': Tensor(N,P,2), 'labels': Tensor(N,),
                           'bboxes': Tensor(N,4), 'polylines_shift': Tensor(N,R,P,2)}}``
            The most recent frame (highest frame_idx) is used.
            Empty scenes store plain Python lists ``[]`` for all fields.

        **loader_typeB** (list-of-dicts):
            ``{frame_idx: [{'gt_label': int, 'p_coords': ndarray, ...}, ...]}``
        """
        # ** NOTE : NOT including background class index **
        out = {c: [] for c in range(self.num_classes)}
        if not isinstance(gt_polylines_item, dict) or len(gt_polylines_item) == 0:
            return out

        _, frame_data = max(gt_polylines_item.items(), key=lambda x: x[0])

        # ── loader_typeA: frame_data is a dict with 'polylines' / 'labels' ──
        if isinstance(frame_data, dict) and 'polylines' in frame_data and 'labels' in frame_data:
            polylines = frame_data['polylines']  # Tensor(N, P, 2)  or  []
            labels    = frame_data['labels']     # Tensor(N,)       or  []

            if not hasattr(polylines, '__len__') or len(polylines) == 0:
                return out
            if not hasattr(labels, '__len__') or len(labels) == 0:
                return out

            pl_np = polylines.cpu().numpy() if torch.is_tensor(polylines) else np.asarray(polylines)
            lb_np = labels.cpu().numpy()    if torch.is_tensor(labels)    else np.asarray(labels)

            for idx in range(len(lb_np)):
                cls_id = int(lb_np[idx])
                if cls_id < 0 or cls_id >= self.num_classes:
                    continue
                pts = self._ensure_pts_xy(pl_np[idx])   # (P, 2)
                if len(pts) < 2:
                    continue
                out[cls_id].append(self._resample_polyline(pts, self.num_sample_pts))
            return out

        return out

    def _extract_pred_by_class(
        self, pred_pl: np.ndarray, pred_sc: np.ndarray
    ) -> Dict[int, Tuple[List[np.ndarray], List[float]]]:
        """Extract predictions per class: {cls_id: (list of (n_pts,2) arrays, list of scores)}.

        Following the Mask2Map evaluation protocol, ALL query predictions are included
        without any score-based or background-based filtering. Each query is assigned to
        its best foreground class. Low-confidence predictions naturally rank at the tail
        of the PR curve and have negligible impact on AP.

        Multi-class (with background): use softmax so probabilities sum to 1 and
        the last class is P(background). No-background setup: use sigmoid.
        """

        # ** NOTE : NOT including background class index **
        out = {c: ([], []) for c in range(self.num_classes)}
        logits = torch.from_numpy(pred_sc.astype(np.float32))

        if self.has_background_class:
            # Multi-class single-label: softmax over all classes (fg + background).
            # Matches decoder ClassificationLoss (F.softmax + cross_entropy).
            probs = logits.softmax(dim=-1).to('cpu').numpy()
            fg_probs = probs[:, :self.num_classes]   # (num_queries, num_fg_classes)
        else:
            fg_probs = logits[:, :self.num_classes].sigmoid().to('cpu').numpy() # (num_queries, num_fg_classes)

        for q in range(pred_pl.shape[0]):

            # Determine the best foreground class and its score
            cls_id = int(np.argmax(fg_probs[q]))
            score = float(fg_probs[q, cls_id]) # best foreground class score

            # No score filtering or background filtering is applied here.
            # All predictions enter the AP computation, matching Mask2Map's protocol.

            # skip queries where the class index is out of range
            if cls_id < 0 or cls_id >= self.num_classes:
                continue

            pts = self._ensure_pts_xy(pred_pl[q])
            if len(pts) < 2:
                continue
            
            resampled = self._resample_polyline(pts, self.num_sample_pts)
            out[cls_id][0].append(resampled)
            out[cls_id][1].append(score)
        return out

    def update(
        self,
        pred_polylines: Union[torch.Tensor, np.ndarray],
        pred_scores: Union[torch.Tensor, np.ndarray],
        gt_polylines: List,
    ):
        """
        Accumulate predictions and ground truth for mAP computation.

        Args:
            pred_polylines: (B, num_queries, num_points, 2)
            pred_scores:    (B, num_queries, num_classes) raw logits.
                            When has_background_class=True the last class is background.
            gt_polylines:   List of length B.  Each element is one of:

                *loader_typeA* (NuScenes tensor format) —
                    ``{frame_idx: {'polylines': Tensor(N,P,2),
                                   'labels':    Tensor(N,),
                                   'bboxes':    Tensor(N,4),
                                   'polylines_shift': Tensor(N,R,P,2)}}``
                    Labels are foreground class indices (0 .. num_fg-1).
                    Empty scenes store plain ``[]`` for all tensor fields.

                *loader_typeB* (list-of-dicts format) —
                    ``{frame_idx: [{'gt_label': int, 'p_coords': ndarray, ...}]}``
                    Labels are foreground class indices (0 .. num_fg-1).
        """
        # predictions including background class
        pred_pl = pred_polylines.detach().cpu().numpy() if torch.is_tensor(pred_polylines) else np.asarray(pred_polylines)
        pred_sc = pred_scores.detach().cpu().numpy() if torch.is_tensor(pred_scores) else np.asarray(pred_scores)
        
        B = pred_pl.shape[0]
        for b in range(B):
            
            # gts_by_cls[cls_id] : a list of polylines, not including background class index, confirmed.
            gts_by_cls = self._extract_gt_by_class(gt_polylines[b]) 

            # preds_by_cls[cls_id][0] : a list of polylines, preds_by_cls[cls_id][1] : a list of scores, not including background class index, confirmed.
            preds_by_cls = self._extract_pred_by_class(pred_pl[b], pred_sc[b]) 
            
            for cls_id in range(self.num_classes):

                # gt and preds for the current class
                gt_lines = gts_by_cls[cls_id]
                pred_lines, pred_scores_list = preds_by_cls[cls_id]
                self._num_gts_by_cls[cls_id] += len(gt_lines)

                if len(pred_lines) == 0 and len(gt_lines) == 0:
                    continue
                if len(pred_lines) == 0:
                    continue

                # stack
                pred_lines = np.stack(pred_lines, axis=0)
                pred_scores_arr = np.array(pred_scores_list, dtype=np.float32)
                gt_arr = np.stack(gt_lines, axis=0) if len(gt_lines) > 0 else np.zeros((0, self.num_sample_pts, 2), dtype=np.float64)

                # calculate tp and fp for each threshold
                self._pred_scores_by_cls[cls_id].append(pred_scores_arr)
                for thr in self.chamfer_thresholds:
                    tp, fp = self._compute_tp_fp(pred_lines, pred_scores_arr, gt_arr, thr)
                    self._tp_by_cls_thr[cls_id][thr].append(tp)
                    self._fp_by_cls_thr[cls_id][thr].append(fp)


    def compute(self) -> Dict[str, float]:
        """
        Compute mAP and per-class AP (averaged over Chamfer thresholds).

        Returns:
            dict with keys: 'mAP', 'AP/drivable', 'AP/divider', ... (per class)
        """
        result = {}
        aps_per_cls = []
        for cls_id in range(self.num_classes):
            scores_list = self._pred_scores_by_cls[cls_id]
            num_gts = self._num_gts_by_cls[cls_id]
            if num_gts == 0:
                result[f'AP/{self.class_names[cls_id]}'] = 0.0
                continue
            if len(scores_list) == 0:
                result[f'AP/{self.class_names[cls_id]}'] = 0.0
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
            result[f'AP/{self.class_names[cls_id]}'] = ap
            aps_per_cls.append(ap)
        result['mAP'] = float(np.mean(aps_per_cls)) if aps_per_cls else 0.0
        return result

    def reset(self):
        """Reset accumulated state."""
        for c in range(self.num_classes):
            self._pred_scores_by_cls[c] = []
            for thr in self.chamfer_thresholds:
                self._tp_by_cls_thr[c][thr] = []
                self._fp_by_cls_thr[c][thr] = []
            self._num_gts_by_cls[c] = 0
