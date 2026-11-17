"""Geometric and photometric data augmentation for instance segmentation.

Geometric transforms operate on the annotation *polygons*, not on rasterized
masks: rotating or rescaling a polygon is exact, whereas transforming an
already rasterized binary mask introduces aliasing along the boundary. The
order used by the training pipeline is therefore

    load polygons -> geometric augmentation (full image) -> crop tile
    -> rasterize instance masks -> photometric augmentation (pixels only).

Geometric functions take and return ``(image, polygons)``, where ``polygons``
is a list of polygons and each polygon a list of ``[x, y]`` pixel coordinates
(floats). Photometric functions only modify the image.
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


# =============================================================================
# Geometric transforms (image and polygons together)
# =============================================================================

def flip_horizontal(image: np.ndarray, polygons: list):
    """Mirror the image and the polygons about the vertical axis."""
    w = image.shape[1]
    out = image[:, ::-1, :].copy()
    out_polys = [[[w - x, y] for x, y in poly] for poly in polygons]
    return out, out_polys


def flip_vertical(image: np.ndarray, polygons: list):
    """Mirror the image and the polygons about the horizontal axis."""
    h = image.shape[0]
    out = image[::-1, :, :].copy()
    out_polys = [[[x, h - y] for x, y in poly] for poly in polygons]
    return out, out_polys


def rotate(image: np.ndarray, polygons: list, angle_deg: float):
    """Rotate about the image centre keeping the canvas size.

    Content rotated outside the canvas is cropped. Uncovered corners are filled
    with the mean image colour so that no artificial black borders are created.
    """
    h, w = image.shape[:2]
    mean_colour = tuple(int(c) for c in image.reshape(-1, image.shape[-1]).mean(axis=0))
    rotated = Image.fromarray(image).rotate(angle_deg, resample=Image.BILINEAR,
                                            expand=False, fillcolor=mean_colour)
    out = np.array(rotated)

    # PIL rotates the image counter-clockwise for a positive angle in screen
    # coordinates (y axis pointing down). The polygon vertices are rotated by
    # the same angle about the same centre so that they stay aligned.
    cx, cy = w / 2.0, h / 2.0
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out_polys = []
    for poly in polygons:
        new_poly = []
        for x, y in poly:
            dx, dy = x - cx, y - cy
            new_poly.append([cx + dx * cos_t + dy * sin_t,
                             cy - dx * sin_t + dy * cos_t])
        out_polys.append(new_poly)
    return out, out_polys


def rescale(image: np.ndarray, polygons: list, factor: float):
    """Isotropic rescaling of the image and the polygons."""
    h, w = image.shape[:2]
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    out = np.array(Image.fromarray(image).resize((nw, nh), Image.BILINEAR))
    out_polys = [[[x * factor, y * factor] for x, y in poly] for poly in polygons]
    return out, out_polys


# =============================================================================
# Photometric transforms (pixels only)
# =============================================================================

def jitter_brightness_contrast(image: np.ndarray, brightness: float,
                               contrast: float) -> np.ndarray:
    img = image.astype(np.float32)
    mean = img.mean()
    img = (img - mean) * contrast + mean
    img = img * brightness
    return np.clip(img, 0, 255).astype(np.uint8)


def jitter_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = np.array(Image.fromarray(image).convert("HSV")).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    rgb = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")
    return np.array(rgb)


def gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return image
    return np.array(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=radius)))


# =============================================================================
# Configuration and combined application
# =============================================================================

@dataclass
class AugConfig:
    """Probabilities and ranges of every augmentation (defaults used in the paper)."""
    # geometric
    p_flip_h: float = 0.5
    p_flip_v: float = 0.5
    p_rotation: float = 0.5
    rotation_range: tuple = (-20.0, 20.0)
    p_scale: float = 0.3
    scale_range: tuple = (0.85, 1.15)
    # photometric
    p_brightness_contrast: float = 0.5
    brightness_range: tuple = (0.8, 1.2)
    contrast_range: tuple = (0.8, 1.2)
    p_saturation: float = 0.3
    saturation_range: tuple = (0.7, 1.3)
    p_noise: float = 0.2
    noise_sigma_range: tuple = (0.0, 10.0)
    p_blur: float = 0.2
    blur_radius_range: tuple = (0.0, 1.5)

    @classmethod
    def disabled(cls) -> "AugConfig":
        """Configuration with every augmentation switched off."""
        return cls(p_flip_h=0, p_flip_v=0, p_rotation=0, p_scale=0,
                   p_brightness_contrast=0, p_saturation=0, p_noise=0, p_blur=0)


def augment_geometric(image: np.ndarray, polygons: list, cfg: AugConfig):
    """Random flips, rotation and rescaling, applied to the full image
    (already normalized to the working scale) before the tile is cropped."""
    if np.random.rand() < cfg.p_flip_h:
        image, polygons = flip_horizontal(image, polygons)
    if np.random.rand() < cfg.p_flip_v:
        image, polygons = flip_vertical(image, polygons)
    if np.random.rand() < cfg.p_rotation:
        image, polygons = rotate(image, polygons, np.random.uniform(*cfg.rotation_range))
    if np.random.rand() < cfg.p_scale:
        image, polygons = rescale(image, polygons, np.random.uniform(*cfg.scale_range))
    return image, polygons


def augment_photometric(image: np.ndarray, cfg: AugConfig) -> np.ndarray:
    """Pixel-level augmentation, applied after cropping the tile (cheaper, and
    it cannot affect the masks)."""
    if np.random.rand() < cfg.p_brightness_contrast:
        b = np.random.uniform(*cfg.brightness_range)
        c = np.random.uniform(*cfg.contrast_range)
        image = jitter_brightness_contrast(image, b, c)
    if np.random.rand() < cfg.p_saturation:
        image = jitter_saturation(image, np.random.uniform(*cfg.saturation_range))
    if np.random.rand() < cfg.p_noise:
        image = gaussian_noise(image, np.random.uniform(*cfg.noise_sigma_range))
    if np.random.rand() < cfg.p_blur:
        image = gaussian_blur(image, np.random.uniform(*cfg.blur_radius_range))
    return image
