"""Image augmentation utilities for camera data preprocessing.

This module provides augmentation functionality for camera images, including
random resizing and cropping, with corresponding updates to camera intrinsic parameters.
"""

from typing import Tuple
import numpy as np
from utils.geometry import *
from PIL import Image as PILImage
from torch import nn
import torch
import random
import os
from torchvision import transforms


class ImageAugmentation:
    """Image augmentation class for camera data.
    
    Applies random resize and crop transformations to images while maintaining
    consistency with camera intrinsic parameters. This is crucial for maintaining
    geometric correctness when projecting between image and 3D space.
    """

    def __init__(self, data_aug_conf):
        """Initialize augmentation with configuration.
        
        Args:
            data_aug_conf: Dictionary containing augmentation parameters:
                - final_dim: (height, width) of final output image
                - resize_lim: (min, max) range for random resize scale (optional)
                - resize_scale: Fixed resize scale if resize_lim not provided
                - crop_offset: Maximum random offset for crop center
        """
        self.data_aug_conf = data_aug_conf

    def sample_augmentation(self):
        """Sample random augmentation parameters (resize and crop).
        
        Returns:
            resize_dims: (width, height) tuple for resized image dimensions
            crop: (x0, y0, x1, y1) tuple defining crop region
        """
        fH, fW = self.data_aug_conf['final_dim']

        # Sample random resize scale
        if 'resize_lim' in self.data_aug_conf and self.data_aug_conf['resize_lim'] is not None:
            # Random resize within specified limits
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
        else:
            # Use fixed resize scale
            resize = self.data_aug_conf['resize_scale']

        # Calculate resized dimensions
        resize_dims = (int(fW * resize), int(fH * resize))
        newW, newH = resize_dims

        # Calculate centered crop position
        crop_h = int((newH - fH) / 2)
        crop_w = int((newW - fW) / 2)

        # Add random offset to crop center for data augmentation
        crop_offset = self.data_aug_conf['crop_offset']
        crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
        crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

        # Define crop region as (x0, y0, x1, y1)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

        return resize_dims, crop

    def __call__(self, image, intrinsic):
        """Apply augmentation to image and update camera intrinsics.
        
        Args:
            image: PIL.Image format, RGB image (3 x H x W)
            intrinsic: torch.Tensor, 3x3 camera intrinsic matrix
        
        Returns:
            image: Augmented PIL.Image
            intrinsic: Updated 3x3 camera intrinsic matrix corresponding to augmented image
        """
        W, H = image.size
        resize_dims, crop = self.sample_augmentation()

        # Update intrinsic parameters to account for resize
        sx = resize_dims[0] / float(W)  # Scale factor in x direction
        sy = resize_dims[1] / float(H)  # Scale factor in y direction

        # Scale intrinsic matrix by resize factors
        intrinsic = scale_intrinsics(intrinsic.unsqueeze(0), sx, sy).squeeze(0)

        # Update principal point (x0, y0) to account for crop offset
        fx, fy, x0, y0 = split_intrinsics(intrinsic.unsqueeze(0))
        new_x0 = x0 - crop[0]  # Adjust principal point x by crop offset
        new_y0 = y0 - crop[1]  # Adjust principal point y by crop offset

        # Reconstruct intrinsic matrix with updated principal point
        pix_T_cam = merge_intrinsics(fx, fy, new_x0, new_y0)
        intrinsic = pix_T_cam.squeeze(0)

        # Apply resize and crop to image
        image = resize_and_crop_image(image, resize_dims, crop)

        return image, intrinsic[:3, :3]

class PhotoMetricDistortionMultiViewImage:
    """Standalone version of PhotoMetricDistortionMultiViewImage.

    This class mirrors the behavior of
    `projects.mmdet3d_plugin.datasets.pipelines.transform_3d.PhotoMetricDistortionMultiViewImage`
    but is completely independent of the MMDetection pipeline registration
    system so that it can be used in isolation.
    """

    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta: int = 18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results: dict) -> dict:
        """Apply photometric distortion to multi-view images.

        Expects ``results`` to contain:
            - 'img': list of images, where each element can be either:
                * np.ndarray with dtype np.float32, BGR order, values in
                  [0, 255] (i.e., uint8 images converted to float32), or
                * PIL.Image in RGB order, with uint8 dynamic range [0, 255],
                  in which case it will be converted internally to float32
                  BGR [0, 255] before applying the distortions.
        """
        from numpy import random  # local import to mirror original behavior

        imgs = results["img"]
        # Track which inputs were originally PIL.Image so we can restore type.
        was_pil = [
            (PILImage is not None and isinstance(im, PILImage.Image))
            for im in imgs
        ]

        new_imgs = []
        for img, is_pil in zip(imgs, was_pil):
            # Accept both np.ndarray and PIL.Image inputs.
            if is_pil:
                # PIL image: RGB uint8 [0, 255] -> numpy float32 BGR [0, 255]
                img_np = np.array(img, dtype=np.float32)
                if img_np.ndim == 2:  # grayscale -> fake 3-channel
                    img_np = np.stack([img_np] * 3, axis=-1)
                # RGB -> BGR
                img = img_np[..., ::-1]
            else:
                assert isinstance(
                    img, np.ndarray
                ), "PhotoMetricDistortion expects images as np.ndarray or PIL.Image."

            # Ensure float32 dtype.
            if img.dtype != np.float32:
                img = img.astype(np.float32)

            # If values are in [0, 1], rescale to [0, 255] to match mmcv behavior.
            img_max = float(img.max()) if img.size > 0 else 0.0
            if img_max <= 1.0 + 1e-3:
                img = img * 255.0

            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img = img + delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img = img * alpha

            # convert color from BGR to HSV (no mmcv dependency)
            img = _bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR (no mmcv dependency)
            img = _hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img = img * alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]

            # Convert back to PIL.Image if this input was PIL.Image
            if is_pil and PILImage is not None:
                img = img[..., ::-1]  # BGR -> RGB
                img = PILImage.fromarray(img.astype(np.uint8))

            new_imgs.append(img)

        results["img"] = new_imgs
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str

class GridMask(nn.Module):
    def __init__(self, use_h, use_w, rotate = 1, offset=False, ratio = 0.5, mode=0, prob = 1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.fp16_enable = False

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch #+ 1.#0.5
    
    def forward(self, x):

        if np.random.rand() > self.prob:
            return x
        n,c,h,w = x.size()

        # Collapse batch+channel so a single 2D mask applies per (n,c) slice.
        x = x.view(-1,h,w)

        # Build a larger mask to avoid empty corners after rotation, then center-crop back.
        hh = int(1.5*h)
        ww = int(1.5*w)
        d = np.random.randint(2, h)

        # Stripe width within each grid cell (clamped to [1, d-1]).
        self.l = min(max(int(d*self.ratio+0.5),1),d-1)
        mask = np.ones((hh, ww), np.float32)

        # Random phase shift of the grid.
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh//d):
                s = d*i + st_h
                t = min(s+self.l, hh)
                mask[s:t,:] *= 0
        if self.use_w:
            for i in range(ww//d):
                s = d*i + st_w
                t = min(s+self.l, ww)
                mask[:,s:t] *= 0
       
        
        # Rotate in PIL space for simplicity, then convert back to numpy.
        r = np.random.randint(self.rotate)
        mask = PILImage.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

        # Match dtype/device, then broadcast to all slices.
        mask = torch.from_numpy(mask).to(x.dtype).cuda()
        if self.mode == 1:
            mask = 1-mask
        mask = mask.expand_as(x)
        if self.offset:
            # Optionally fill masked-out regions with random noise instead of zeros.
            offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).to(x.dtype).cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask 
        
        return x.view(n,c,h,w)

def _bgr2hsv(img: np.ndarray) -> np.ndarray:
    """Convert BGR image (float32, 0–255) to HSV (H in degrees 0–360, S,V 0–1)."""
    img = img.astype(np.float32)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc / 255.0
    deltac = maxc - minc

    s = np.zeros_like(maxc, dtype=np.float32)
    non_zero = maxc != 0
    s[non_zero] = deltac[non_zero] / maxc[non_zero]

    h = np.zeros_like(maxc, dtype=np.float32)
    mask = deltac != 0

    # Red is max
    mask_r = (maxc == r) & mask
    h[mask_r] = (g[mask_r] - b[mask_r]) / deltac[mask_r]

    # Green is max
    mask_g = (maxc == g) & mask
    h[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / deltac[mask_g]

    # Blue is max
    mask_b = (maxc == b) & mask
    h[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / deltac[mask_b]

    h = (h / 6.0) % 1.0  # normalize to [0,1)
    h = h * 360.0        # degrees

    hsv = np.empty_like(img, dtype=np.float32)
    hsv[..., 0] = h
    hsv[..., 1] = s
    hsv[..., 2] = v
    return hsv


def _hsv2bgr(img: np.ndarray) -> np.ndarray:
    """Convert HSV (H in degrees 0–360, S,V 0–1) to BGR image (float32, 0–255)."""
    h = img[..., 0] / 60.0  # sector 0 to 5
    s = img[..., 1]
    v = img[..., 2]

    c = v * s
    x = c * (1 - np.abs(h % 2 - 1))
    m = v - c

    z = np.zeros_like(h, dtype=np.float32)

    b = np.empty_like(h, dtype=np.float32)
    g = np.empty_like(h, dtype=np.float32)
    r = np.empty_like(h, dtype=np.float32)

    h0 = (0 <= h) & (h < 1)
    h1 = (1 <= h) & (h < 2)
    h2 = (2 <= h) & (h < 3)
    h3 = (3 <= h) & (h < 4)
    h4 = (4 <= h) & (h < 5)
    h5 = (5 <= h) & (h < 6)

    r[h0], g[h0], b[h0] = c[h0], x[h0], z[h0]
    r[h1], g[h1], b[h1] = x[h1], c[h1], z[h1]
    r[h2], g[h2], b[h2] = z[h2], c[h2], x[h2]
    r[h3], g[h3], b[h3] = z[h3], x[h3], c[h3]
    r[h4], g[h4], b[h4] = x[h4], z[h4], c[h4]
    r[h5], g[h5], b[h5] = c[h5], z[h5], x[h5]

    r = (r + m) * 255.0
    g = (g + m) * 255.0
    b = (b + m) * 255.0

    out = np.empty_like(img, dtype=np.float32)
    out[..., 0] = b
    out[..., 1] = g
    out[..., 2] = r
    return out
