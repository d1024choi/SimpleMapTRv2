import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from utils.verification import draw_traj_on_topview
import torch.nn.functional as F
import os
from textwrap import fill

# many colors from
COLORS = {
    'drivable': (110, 110, 110),       # dark grey
    'divider': (0, 0, 255),            # red
    'ped_crossing': (251, 154, 153),   # light purple
    'walkway': (0, 128, 0),            # dark green
    'carpark_area': (255, 127, 0),     # light blue
    'stop_line': (253, 191, 111),
    'vehicle': (0, 158, 255),          # orange
    'pedestrian': (230, 0, 0),         # blue
    'nothing': (200, 200, 200)         # light grey
}


# BEV part -------------------------------
def colorize(x, colormap=None):
    """
    x: (h w) np.uint8 0-255
    colormap
    """
    try:
        return (255 * get_cmap(colormap)(x)[..., :3]).astype(np.uint8)
    except:
        pass

    if x.dtype == np.float32:
        x = (255 * x).astype(np.uint8)

    if colormap is None:
        return x[..., None].repeat(3, 2)

    return cv2.applyColorMap(x, getattr(cv2, f'COLORMAP_{colormap.upper()}'))


def get_colors(semantics):
    return np.array([COLORS[s] for s in semantics], dtype=np.uint8)


def to_image(x):
    return (255 * x).byte().cpu().numpy().transpose(1, 2, 0)


def greyscale(x):
    return (255 * x.repeat(3, 2)).astype(np.uint8)


def resize(src, dst=None, shape=None, idx=0):
    if dst is not None:
        ratio = dst.shape[idx] / src.shape[idx]
    elif shape is not None:
        ratio = shape[idx] / src.shape[idx]

    width = int(ratio * src.shape[1])
    height = int(ratio * src.shape[0])

    return cv2.resize(src, (width, height), interpolation=cv2.INTER_CUBIC)


def visualize_bev_coord_3d(bev_coord_3d, save_path=None, show_plot=True, subsample=None, figsize=(15, 12)):
    """
    Visualize BEV 3D coordinates.
    
    Args:
        bev_coord_3d: Tensor or numpy array with shape (D, 3, H, W)
                     where D is number of depth levels, 3 is (x, y, z), H and W are spatial dimensions
        save_path: Optional path to save the figure (e.g., 'bev_coord_3d.png')
        show_plot: Whether to display the plot (default: True)
        subsample: Optional subsampling factor for faster visualization (e.g., 2 means every 2nd point)
        figsize: Figure size tuple (width, height) in inches
    
    Returns:
        matplotlib figure object
    """
    # Convert to numpy if tensor
    if isinstance(bev_coord_3d, torch.Tensor):
        bev_coord_3d = bev_coord_3d.detach().cpu().numpy()
    
    D, _, H, W = bev_coord_3d.shape
    
    # Extract coordinates
    x = bev_coord_3d[:, 0, :, :]  # D H W
    y = bev_coord_3d[:, 1, :, :]  # D H W
    z = bev_coord_3d[:, 2, :, :]  # D H W
    
    # Subsample if requested
    if subsample is not None:
        x = x[::subsample, ::subsample, ::subsample]
        y = y[::subsample, ::subsample, ::subsample]
        z = z[::subsample, ::subsample, ::subsample]
    
    # Flatten for plotting
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # 3D scatter plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    scatter = ax1.scatter(x_flat, y_flat, z_flat, c=z_flat, cmap='viridis', 
                         alpha=0.3, s=1, edgecolors='none')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D BEV Coordinates')
    plt.colorbar(scatter, ax=ax1, label='Z (m)')
    
    # Top view (X-Y projection)
    ax2 = fig.add_subplot(2, 2, 2)
    scatter2 = ax2.scatter(x_flat, y_flat, c=z_flat, cmap='viridis', 
                          alpha=0.3, s=1, edgecolors='none')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (X-Y Projection)')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label='Z (m)')
    
    # Side view (X-Z projection)
    ax3 = fig.add_subplot(2, 2, 3)
    scatter3 = ax3.scatter(x_flat, z_flat, c=y_flat, cmap='plasma', 
                          alpha=0.3, s=1, edgecolors='none')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (X-Z Projection)')
    plt.colorbar(scatter3, ax=ax3, label='Y (m)')
    
    # Front view (Y-Z projection)
    ax4 = fig.add_subplot(2, 2, 4)
    scatter4 = ax4.scatter(y_flat, z_flat, c=x_flat, cmap='coolwarm', 
                          alpha=0.3, s=1, edgecolors='none')
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Front View (Y-Z Projection)')
    plt.colorbar(scatter4, ax=ax4, label='X (m)')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


class BaseViz:
    """BEV visualization class that only visualizes classes in targets."""

    def __init__(self, targets, label_indices, Thresholds):
        self.targets = targets
        self.label_indices = label_indices
        self.Thresholds = Thresholds
        
        # Generate colors only for classes in targets (plus 'nothing' for background)
        color_order = self.targets + ['nothing']
        self.colors = get_colors(color_order)

    def return_bev_map(self, label, target='vehicle'):
        """Extract BEV map for a specific target class.
        
        Args:
            label: BEV label tensor (b x 12 x h x w)
            target: Target class name
        
        Returns:
            BEV map tensor (b x 1 x h x w)
        """
        label = [label[:, idx].max(dim=1, keepdim=True).values for idx in self.label_indices[target]]
        return torch.cat(label, dim=1)

    def __call__(self, batch, pred):
        """Generate visualization combining images, GT BEV, and predicted BEV.
        
        Args:
            batch: Batch dictionary with 'bev' and 'image' keys
            pred: Prediction dictionary
        
        Returns:
            List of visualization images (one per batch item)
        """
        bev_gt = self.vis_gt(batch['bev'])
        bev_pred = self.vis_pred(pred)

        output = []
        batch_size = batch['image'].size(0)
        for b in range(batch_size):
            imgs = [to_image(batch['image'][b][c]) for c in range(batch['image'].size(1))]
            imgs = np.vstack((np.hstack(imgs[:3]), np.hstack(imgs[3:])))
            imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
            imgs = cv2.resize(imgs, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            h, _, _ = imgs.shape
            gt = cv2.resize(bev_gt[b], dsize=(h, h), interpolation=cv2.INTER_NEAREST)
            pr = cv2.resize(bev_pred[b], dsize=(h, h), interpolation=cv2.INTER_NEAREST)
            output.append(np.hstack((imgs, gt, pr)))

        return output

    def vis_gt(self, bev):
        """Visualize ground truth BEV labels.
        
        Args:
            bev: BEV label tensor (b x 12 x h x w)
        
        Returns:
            List of RGB visualization images (h x w x 3) for each batch item
        """
        # Extract maps only for classes in targets
        maps = []
        for target in self.targets:
            if target in self.label_indices:
                target_map = self.return_bev_map(bev, target=target).permute(0, 2, 3, 1)
                maps.append(target_map)
        
        if not maps:
            return [np.zeros((bev.shape[2], bev.shape[3], 3), dtype=np.uint8) for _ in range(bev.shape[0])]
        
        bev_all = torch.cat(maps, dim=-1).numpy()  # b x h x w x num_targets
        batch, h, w, c = bev_all.shape
        
        output = []
        empty_color = np.uint8(COLORS['nothing'])[None, None]  # 1 1 3
        
        for b in range(batch):
            bev_cur = bev_all[b]  # h w c
            
            # Prioritize higher class labels (later classes in targets list)
            eps = (1e-5 * np.arange(c))[None, None]  # 1 1 c
            idx = (bev_cur + eps).argmax(axis=-1)  # h w
            val = np.take_along_axis(bev_cur, idx[..., None], -1).squeeze(-1)  # h w
            
            # Apply colors: val * color + (1 - val) * background
            result = (val[..., None] * self.colors[idx]) + ((1 - val[..., None]) * empty_color)
            output.append(np.uint8(result))

        return output

    def vis_pred(self, pred):
        """Visualize predicted BEV segmentation.
        
        Args:
            pred: Prediction dictionary with class predictions
        
        Returns:
            List of RGB visualization images (h x w x 3) for each batch item
        """
        batch_size, _, h, w = pred[self.targets[0]][0].size()
        
        # Extract predictions only for classes in targets
        maps = []
        for target in self.targets:
            if target in pred and target in self.Thresholds:
                thresholded = (F.sigmoid(pred[target][0]).permute(0, 2, 3, 1) > self.Thresholds[target])
                target_map = thresholded.detach().to('cpu').numpy().astype(np.float32)
                maps.append(target_map)
        
        if not maps:
            return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(batch_size)]
        
        bev_all = np.concatenate(maps, axis=-1)  # b x h x w x num_targets
        batch, h, w, c = bev_all.shape
        
        output = []
        empty_color = np.uint8(COLORS['nothing'])[None, None]  # 1 1 3
        
        for b in range(batch):
            bev_cur = bev_all[b]  # h w c
            
            # Prioritize higher class labels (later classes in targets list)
            eps = (1e-5 * np.arange(c))[None, None]  # 1 1 c
            idx = (bev_cur + eps).argmax(axis=-1)  # h w
            val = np.take_along_axis(bev_cur, idx[..., None], -1).squeeze(-1)  # h w
            
            # Apply colors: val * color + (1 - val) * background
            result = (val[..., None] * self.colors[idx]) + ((1 - val[..., None]) * empty_color)
            output.append(np.uint8(result))

        return output


class VLMVisualizer:
    """
    Visualizes VLM (Vision-Language Model) results including camera images and text outputs.
    
    Usage:
        vis = VLMVisualizer(save_dir='./results', ckp_id=65)
        vis.visualize(images, question, gt_answer, gen_answer, batch_idx=0, save=True)
    """
    
    def __init__(self, save_dir=None, ckp_id=None, figsize=(20, 12), dpi=150):
        """
        Initialize the VLM visualizer.
        
        Args:
            save_dir: Directory to save visualizations (if None, will show interactively)
            ckp_id: Checkpoint ID for organizing saved files
            figsize: Figure size (width, height) in inches
            dpi: Resolution for saved figures
        """
        self.save_dir = save_dir
        self.ckp_id = ckp_id
        self.figsize = figsize
        self.dpi = dpi
        self.camera_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
                            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        
        # Create save directory if needed
        if self.save_dir is not None and self.ckp_id is not None:
            self.vis_dir = os.path.join(self.save_dir, f'vis_ckp{self.ckp_id}')
            os.makedirs(self.vis_dir, exist_ok=True)
        else:
            self.vis_dir = None
    
    def visualize(self, images, question, gt_answer, gen_answer, batch_idx=0, save=True, show=False):
        """
        Create and save/show visualization of images and texts.
        
        Args:
            images: numpy array of shape (b, 6, 3, H, W) or (6, 3, H, W) - images in range [0, 1]
            question: str - the question text
            gt_answer: str - ground truth answer text
            gen_answer: str - generated answer text
            batch_idx: int - batch index for naming saved files
            save: bool - whether to save the figure
            show: bool - whether to display the figure interactively
        """
        # Handle input shape: if 4D (b, 6, 3, H, W), take first batch
        if images.ndim == 5:
            batch_images = images[0]  # [6, 3, H, W]
        elif images.ndim == 4:
            batch_images = images  # [6, 3, H, W]
        else:
            raise ValueError(f"Expected images shape (b, 6, 3, H, W) or (6, 3, H, W), got {images.shape}")
        
        # Create figure with subplots: 3 rows (2 for images, 1 for text)
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.2)
        
        # Display 6 camera images in 2x3 grid
        for cam_idx in range(6):
            row = cam_idx // 3
            col = cam_idx % 3
            ax = fig.add_subplot(gs[row, col])
            
            # Convert from CHW to HWC and clip to [0, 1] range
            img = np.transpose(batch_images[cam_idx], (1, 2, 0))  # [H, W, 3]
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.set_title(self.camera_names[cam_idx], fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Add text panel at the bottom spanning all 3 columns
        text_ax = fig.add_subplot(gs[2, :])
        text_ax.axis('off')
        
        # Format text with wrapping
        text_content = f"""
{fill(f"❓ Question: {question}", width=100)}

{fill(f"✅ Ground Truth Answer: {gt_answer}", width=100)}

{fill(f"🤖 Generated Answer: {gen_answer}", width=100)}
"""
        
        text_ax.text(0.05, 0.95, text_content, transform=text_ax.transAxes,
                   fontsize=11, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'VLM Results - Batch {batch_idx}', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save or show
        if save and self.vis_dir is not None:
            save_path = os.path.join(self.vis_dir, f'batch{batch_idx:04d}_vlm.png')
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        elif show:
            plt.show()
            plt.close()
        else:
            plt.close()
    
    def visualize_batch(self, images, questions, gt_answers, gen_answers, batch_start_idx=0, save=True, show=False):
        """
        Visualize multiple samples in a batch.
        
        Args:
            images: numpy array of shape (b, 6, 3, H, W)
            questions: list of question strings
            gt_answers: list of ground truth answer strings
            gen_answers: list of generated answer strings
            batch_start_idx: starting index for file naming
            save: bool - whether to save figures
            show: bool - whether to display figures interactively
        """
        batch_size = len(questions)
        for i in range(batch_size):
            # Extract images for this sample
            sample_images = images[i:i+1] if images.ndim == 5 else images
            self.visualize(
                images=sample_images,
                question=questions[i],
                gt_answer=gt_answers[i],
                gen_answer=gen_answers[i],
                batch_idx=batch_start_idx + i,
                save=save,
                show=show
            )


class PolylineVisualizer:
    """
    Visualizes predicted and ground truth polylines on BEV map alongside the
    six surround-view camera images, producing a single composite figure per
    sample that matches the ``__TEST__.py`` style:

    ::

        ┌─────────────┬─────────────┬─────────────┐
        │ CAM_FRONT_L │  CAM_FRONT  │ CAM_FRONT_R │
        ├─────────────┼─────────────┼─────────────┤
        │  CAM_BACK_L │  CAM_BACK   │ CAM_BACK_R  │
        ├──────────────────┬──────────────────┤
        │   GT BEV (plt)   │  Pred BEV (plt)  │
        └──────────────────┴──────────────────┘

    BEV maps use ``matplotlib`` with ``pc_range``-based axes and per-class
    matplotlib colors (e.g. ``'orange'``, ``'b'``, ``'g'``), matching
    ``__TEST__.py``'s ``colors_plt``.

    **Coordinate convention:** Polyline points use ``(x, y) = (right, backward)``
    in the ego frame. ``pc_range`` is ``[xmin, ymin, zmin, xmax, ymax, zmax]``,
    i.e. ``[-w, -h, -z, w, h, z]`` for right/backward/z extents. The BEV plot is
    drawn **as if** ``(left, forward)`` with ``left = -right``, ``forward = -backward``:
    horizontal = left, vertical = forward, forward up, and ``invert_xaxis`` so
    positive left lies to the left on screen.

    Supports two GT formats (auto-detected):
    - loader_typeA: ``{frame_idx: {'polylines': Tensor(N,P,2), 'labels': Tensor(N,)}}``
    - loader_typeB: ``{frame_idx: [{'p_coords': ndarray, 'gt_label': int, 'layer_type': str}, ...]}``
    """

    CAM_ORDER = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT']
    # Input tensor order: [front, front_right, front_left, back, back_left, back_right]
    # Map input index -> display index so that display order matches CAM_ORDER
    INPUT_TO_DISPLAY_ORDER = [2, 0, 1, 4, 3, 5]  # front_left->0, front->1, front_right->2, back_left->3, back->4, back_right->5

    def __init__(self, cfg, score_threshold=0.25, gt_thickness=2, pred_thickness=2,
                 use_bg_class=False, pc_range=None, class_names=None,
                 class_colors_plt=None, car_icon_path=None, dpi=150,
                 # legacy kwargs kept for backward compat (ignored)
                 gt_polyline_color=None, gt_polygon_color=None, pred_color=None,
                 class_colors=None):
        """
        Args:
            cfg: Configuration dict.  Uses ``cfg['nuscenes']['pc_range']`` and
                ``cfg['nuscenes']['bev']`` when present.
            score_threshold: Min foreground confidence to draw a prediction.
            gt_thickness / pred_thickness: matplotlib ``linewidth`` for GT / pred.
            use_bg_class: If True the last class index is background (softmax).
            pc_range: Optional ``[xmin, ymin, zmin, xmax, ymax, zmax]`` for
                right, backward, and z (metres), typically ``[-w, -h, -z, w, h, z]``.
            class_names: e.g. ``['divider', 'ped_crossing', 'boundary']``.
            class_colors_plt: **Matplotlib** color names, one per class,
                matching ``__TEST__.py``'s ``colors_plt``
                (default: ``['orange', 'b', 'g']``).
            car_icon_path: Path to the ego-car icon PNG (``'./figs/lidar_car.png'``).
                Set ``None`` to skip.
            dpi: Resolution of saved matplotlib BEV maps.
        """
        # pc_range
        if pc_range is not None:
            pr = pc_range
        else:
            pr = cfg.get('nuscenes', {}).get('pc_range')
        
        if pr is not None and len(pr) >= 6:
            self.pc_range = [float(v) for v in pr]
        else:
            # default: [-w, -h, -z, w, h, z] (right, backward, z)
            self.pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.score_threshold = score_threshold
        self.gt_thickness = gt_thickness
        self.pred_thickness = pred_thickness
        self.use_bg_class = use_bg_class
        self.class_names = list(class_names) if class_names else []
        self.dpi = dpi

        # Matplotlib colors per class (same convention as __TEST__.py colors_plt)
        if class_colors_plt is not None:
            self.colors_plt = list(class_colors_plt)
        else:
            self.colors_plt = ['orange', 'b', 'g']
        while len(self.colors_plt) < len(self.class_names):
            self.colors_plt.append('m')

        # Car icon (optional)
        self.car_img = None
        if car_icon_path is not None and os.path.isfile(car_icon_path):
            self.car_img = ImageOps.flip(Image.open(car_icon_path))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_gt(self, gt_polylines_item):
        """Return ``(polylines_np, labels_np)`` — both numpy, or (None, None) if empty."""
        if not isinstance(gt_polylines_item, dict) or len(gt_polylines_item) == 0:
            return None, None
        _, frame_data = max(gt_polylines_item.items(), key=lambda x: x[0])
        if isinstance(frame_data, dict) and 'polylines' in frame_data and 'labels' in frame_data:
            pl = frame_data['polylines']
            lb = frame_data['labels']
            if not hasattr(pl, '__len__') or len(pl) == 0:
                return None, None
            pl_np = pl.cpu().numpy() if torch.is_tensor(pl) else np.asarray(pl)
            lb_np = lb.cpu().numpy() if torch.is_tensor(lb) else np.asarray(lb)
            return pl_np, lb_np
        if isinstance(frame_data, (list, tuple)):
            pls, lbs = [], []
            for p in frame_data:
                pc = p.get('p_coords', None)
                gl = p.get('gt_label', None)
                if pc is None or gl is None:
                    continue
                pc = pc.cpu().numpy() if torch.is_tensor(pc) else np.asarray(pc, dtype=np.float64)
                if pc.ndim == 2 and pc.shape[0] == 2:
                    pc = pc.T
                pls.append(pc)
                lbs.append(int(gl))
            if len(pls) == 0:
                return None, None
            return pls, np.array(lbs)
        return None, None

    def _render_bev_matplotlib(self, polylines, labels, title, is_pred=False):
        """Render a BEV map to an RGB numpy array, __TEST__.py style.

        Data ``(x, y) = (right, backward)`` with ``pc_range`` = ``[-w,-h,-z,w,h,z]``,
        plotted as **(left, forward)** via ``left = -right``, ``forward = -backward``:
        horizontal = left, vertical = forward (ahead up); ``invert_xaxis`` for +left leftward.
        """
        import io
        fig = plt.figure(figsize=(2, 4))
        ax = plt.gca()
        # horizontal = right (x), vertical = backward (y)
        ax.set_xlim(self.pc_range[0], self.pc_range[3])
        ax.set_ylim(self.pc_range[1], self.pc_range[4])
        ax.invert_yaxis()   # ahead (ymin / -h) at top, behind (ymax / +h) at bottom
        ax.axis('off')

        if polylines is not None and labels is not None:
            if isinstance(polylines, np.ndarray):
                for idx in range(len(labels)):
                    cls_id = int(labels[idx])
                    color = self.colors_plt[cls_id] if 0 <= cls_id < len(self.colors_plt) else 'm'
                    pts = polylines[idx]
                    if pts.ndim == 2 and pts.shape[0] == 2:
                        pts = pts.T
                    right, backward = pts[:, 0], pts[:, 1]   # (x, y) = (right, backward)
                    lw = self.pred_thickness if is_pred else self.gt_thickness
                    plt.plot(right, backward, color=color, linewidth=lw, alpha=0.8, zorder=-1)
                    plt.scatter(right, backward, color=color, s=2, alpha=0.8, zorder=-1)
            elif isinstance(polylines, list):
                for pts, cls_id in zip(polylines, labels):
                    if pts.ndim == 2 and pts.shape[0] == 2:
                        pts = pts.T
                    color = self.colors_plt[int(cls_id)] if 0 <= int(cls_id) < len(self.colors_plt) else 'm'
                    right, backward = pts[:, 0], pts[:, 1]   # (x, y) = (right, backward)
                    lw = self.pred_thickness if is_pred else self.gt_thickness
                    plt.plot(right, backward, color=color, linewidth=lw, alpha=0.8, zorder=-1)
                    plt.scatter(right, backward, color=color, s=2, alpha=0.8, zorder=-1)

        if self.car_img is not None:
            # extent: [right_min, right_max, backward_bottom, backward_top] (matches inverted y)
            plt.imshow(self.car_img, extent=[-1.5, 1.5, 1.2, -1.2])

        plt.title(title, fontsize=6, color='white', pad=2)
        fig.patch.set_facecolor('black')
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches='tight', format='png', dpi=self.dpi,
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        return img

    @staticmethod
    def _fig_to_array(fig, dpi=150):
        """Rasterize a matplotlib figure to an RGB numpy array."""
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def visualize(self, pred_polylines, pred_scores, gt_polylines, images=None):
        """
        Produce one composite image per batch sample: surround-view cameras
        on top, GT BEV + Pred BEV on the bottom (``__TEST__.py`` layout).

        Args:
            pred_polylines: ``(B, Q, P, 2)`` in BEV metres (denormalised), ``(x,y)=(right, backward)``.
            pred_scores:    ``(B, Q, C)`` logits.
            gt_polylines:   List of length B (loader_typeA or loader_typeB format).
            images:         Optional ``(B, n_cam, C, H, W)`` uint8 camera tensor.
                            When ``None``, the surround-view row is omitted and only
                            the two BEV maps are returned side-by-side.

        Returns:
            List of BGR ``np.ndarray`` images, one per batch item.
        """
        batch_size = pred_polylines.size(0)
        vis_images = []

        for bb in range(batch_size):
            pred_pl = pred_polylines[bb]    # (Q, P, 2)
            pred_sc = pred_scores[bb]       # (Q, C)

            # ── Extract GT ────────────────────────────────────────────
            gt_pl, gt_lb = self._extract_gt(gt_polylines[bb]) # (G, P, 2), (G,)
            

            # ── Filter predictions ────────────────────────────────────
            num_queries = pred_pl.shape[0]
            num_classes = pred_sc.shape[1]
            n_fg = num_classes - 1 if self.use_bg_class else num_classes
            keep_pts, keep_labels = [], []
            for q in range(num_queries):
                if self.use_bg_class:
                    probs = torch.softmax(pred_sc[q].float(), dim=-1)
                    fg_probs = probs[:n_fg]
                    bg_prob = probs[n_fg].item()
                else:
                    fg_probs = torch.sigmoid(pred_sc[q, :n_fg].float())
                    bg_prob = 0.0
                best_cls = int(torch.argmax(fg_probs).item())
                score_val = fg_probs[best_cls].item()
                if bg_prob > score_val or score_val < self.score_threshold:
                    continue
                keep_pts.append(pred_pl[q].cpu().numpy())
                keep_labels.append(best_cls)

            if len(keep_pts) > 0:
                pred_np = np.stack(keep_pts)
                pred_lb = np.array(keep_labels)
            else:
                pred_np, pred_lb = None, None

            # ── BEV maps (matplotlib, __TEST__.py style) ──────────────
            gt_bev = self._render_bev_matplotlib(gt_pl, gt_lb, 'GT MAP')
            pred_bev = self._render_bev_matplotlib(pred_np, pred_lb, 'PRED MAP', is_pred=True)

            # Resize both BEV maps to same height
            target_h = max(gt_bev.shape[0], pred_bev.shape[0])
            if gt_bev.shape[0] != target_h:
                ratio = target_h / gt_bev.shape[0]
                gt_bev = cv2.resize(gt_bev, (int(gt_bev.shape[1] * ratio), target_h))
            if pred_bev.shape[0] != target_h:
                ratio = target_h / pred_bev.shape[0]
                pred_bev = cv2.resize(pred_bev, (int(pred_bev.shape[1] * ratio), target_h))
            bev_row = np.concatenate([gt_bev, pred_bev], axis=1)  # (H, W_gt+W_pred, 3)

            # ── Surround-view camera strip ────────────────────────────
            if images is not None:
                cam_imgs = images[bb]   # (n_cam, C, H, W)
                if torch.is_tensor(cam_imgs):
                    cam_imgs = cam_imgs.cpu()
                n_cam = cam_imgs.shape[0]

                # Build list in input order, swap R↔B (input is RGB; cv2 expects BGR)
                cam_list_input = []
                for ci in range(n_cam):
                    im = cam_imgs[ci]
                    if torch.is_tensor(im):
                        im = im.permute(1, 2, 0).numpy()
                    if im.dtype != np.uint8:
                        im = im.astype(np.uint8)
                    im = im[:, :, ::-1]   # RGB -> BGR (swap blue and red channels)
                    cam_list_input.append(im)

                # Reorder to display order: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT
                cam_list = []
                for display_idx in range(6):
                    input_idx = self.INPUT_TO_DISPLAY_ORDER[display_idx]
                    if input_idx < len(cam_list_input):
                        cam_list.append(cam_list_input[input_idx])
                    elif cam_list_input:
                        cam_list.append(np.zeros_like(cam_list_input[0]))
                while len(cam_list) < 6 and cam_list:
                    cam_list.append(np.zeros_like(cam_list[0]))

                # Flip back-facing cameras (BACK_LEFT, BACK, BACK_RIGHT) so they face forward in the composite
                for i in (3, 4, 5):
                    if i < len(cam_list) and cam_list[i].size > 0:
                        cam_list[i] = np.fliplr(cam_list[i])

                # Compose 2×3 grid (FL F FR / BL B BR)
                row1 = cv2.hconcat(cam_list[:3])
                row2 = cv2.hconcat(cam_list[3:6])
                cam_grid = cv2.vconcat([row1, row2])  # (2*Hcam, 3*Wcam, 3)

                # Resize camera grid width to match BEV row width
                bev_w = bev_row.shape[1]
                cam_ratio = bev_w / cam_grid.shape[1]
                cam_grid = cv2.resize(cam_grid, (bev_w, int(cam_grid.shape[0] * cam_ratio)))

                composite = cv2.vconcat([cam_grid, bev_row])
            else:
                composite = bev_row

            vis_images.append(composite)
        return vis_images


def visualize_polyline_on_bev(bev, polylines, color=(0, 0, 255), thickness=2, pc_range=[-30, -15, -2, 30, 15, 2], map_size=(60, 30)):
    '''
    bev: (H, W, 3) numpy array or None
    polylines: (M, N, 2) numpy array or tensor
    '''

        
    # NuScenes convention: x = forward, y = left, z = up
    # BEV image layout:  top = forward (+x),  left = left (+y)


    if isinstance(polylines, np.ndarray):
        polylines = torch.from_numpy(polylines)

    if (bev is None):
        map_size_r = map_size[0]
        map_size_c = map_size[1]
        bev = np.zeros(shape=(map_size_r, map_size_c, 3))
    else:
        map_size_r = bev.shape[0]
        map_size_c = bev.shape[1]
        
    x_range = (pc_range[0], pc_range[3])
    y_range = (pc_range[1], pc_range[4])
    axis_range_y = y_range[1] - y_range[0]
    axis_range_x = x_range[1] - x_range[0]
    scale_y = float(map_size_c - 1) / axis_range_y
    scale_x = float(map_size_r - 1) / axis_range_x
    
    
    for i in range(polylines.shape[0]):
        # yx = polylines[i] # (seq_len, 2)
        # xy = yx.flip(dims=[1]).numpy() # col 0 = x (forward), col 1 = y (left)
        xy = polylines[i].numpy() # (seq_len, 2)

        seq_len = xy.shape[0]
        # y (left) → col: +y_max (leftmost) lands at col 0 (left image edge)
        col_img = -(xy[:, 1] * scale_y).astype(np.int32)
        # x (forward) → row: +x_max (furthest forward) lands at row 0 (top image edge)
        row_img = -(xy[:, 0] * scale_x).astype(np.int32)

        col_img += int(np.trunc(y_range[1] * scale_y))
        row_img += int(np.trunc(x_range[1] * scale_x))

        # pts layout for cv2: (col, row) = (horizontal, vertical)
        pts = np.concatenate([col_img.reshape(seq_len, 1), row_img.reshape(seq_len, 1)], axis=-1)
        cv2.polylines(bev, [pts[:-1].reshape((-1, 1, 2))], isClosed=False, color=color, thickness=2)
    # cv2.imwrite('./line_drawings_verif.png', img.astype(np.uint8))

    return bev

def visualize_points_on_bev(bev, polylines, color=(0, 0, 255), thickness=2, pc_range=[-30, -15, -2, 30, 15, 2], map_size=(60, 30)):
    '''
    bev: (H, W, 3) numpy array or None
    polylines: (M, N, 2) numpy array or tensor
    '''

        
    # NuScenes convention: x = forward, y = left, z = up
    # BEV image layout:  top = forward (+x),  left = left (+y)


    if isinstance(polylines, np.ndarray):
        polylines = torch.from_numpy(polylines)

    if (bev is None):
        map_size_r = map_size[0]
        map_size_c = map_size[1]
        bev = np.zeros(shape=(map_size_r, map_size_c, 3))
    else:
        map_size_r = bev.shape[0]
        map_size_c = bev.shape[1]
        
    x_range = (pc_range[0], pc_range[3])
    y_range = (pc_range[1], pc_range[4])
    axis_range_y = y_range[1] - y_range[0]
    axis_range_x = x_range[1] - x_range[0]
    scale_y = float(map_size_c - 1) / axis_range_y
    scale_x = float(map_size_r - 1) / axis_range_x
    
    
    for i in range(polylines.shape[0]):
        # yx = polylines[i] # (seq_len, 2)
        # xy = yx.flip(dims=[1]).numpy() # col 0 = x (forward), col 1 = y (left)
        xy = polylines[i].numpy() # (seq_len, 2)

        seq_len = xy.shape[0]
        # y (left) → col: +y_max (leftmost) lands at col 0 (left image edge)
        col_img = -(xy[:, 1] * scale_y).astype(np.int32)
        # x (forward) → row: +x_max (furthest forward) lands at row 0 (top image edge)
        row_img = -(xy[:, 0] * scale_x).astype(np.int32)

        col_img += int(np.trunc(y_range[1] * scale_y))
        row_img += int(np.trunc(x_range[1] * scale_x))

        # pts layout for cv2: (col, row) = (horizontal, vertical)
        pts = np.concatenate([col_img.reshape(seq_len, 1), row_img.reshape(seq_len, 1)], axis=-1)
        for pt in pts:
            cv2.circle(bev, (pt[0], pt[1]), 3, color, -1)

    return bev    