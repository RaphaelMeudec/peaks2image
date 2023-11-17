import torch
import pkg_resources

from .model import Peaks2Image, MAX_NUMBER_OF_PEAKS, PEAKS2IMAGE_PARAMS

peaks2image = Peaks2Image(**PEAKS2IMAGE_PARAMS)
peaks2image.load_state_dict(
    torch.load(
        pkg_resources.resource_filename(__name__, "assets/peaks2image.pt")
    )
)


__all__ = ["peaks2image"]
