"""cIEF-Net: instance segmentation of fungal conidia with a frozen DINOv3
encoder, DoRA adapters, a star-convex head and polar harmonic contour
refinement."""

__version__ = "1.0.0"
__all__ = ["InferenceConfig", "build_cief_net", "build_model", "count_parameters"]

from .inference import InferenceConfig  # noqa: F401
from .model import build_cief_net, build_model, count_parameters  # noqa: F401
