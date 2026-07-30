import math
from dataclasses import dataclass

from PIL import Image, ImageOps


VIEWS = {"full", "v1", "v2", "v3", "v4"}
CROP_FRACTIONS = {"v2": 0.50, "v3": 0.75}
BORDER_PIXELS = 28


@dataclass(frozen=True)
class ViewGeometry:
    view_id: str
    original_size: tuple[int, int]
    view_size: tuple[int, int]
    offset_x: float
    offset_y: float
    center_fallback: bool = False

    def view_to_original_normalized(self, x: float, y: float) -> tuple[float, float]:
        width, height = self.original_size
        return (x + self.offset_x) / width, (y + self.offset_y) / height


@dataclass(frozen=True)
class GeneratedView:
    image: Image.Image
    geometry: ViewGeometry


def _validated_center(center, width, height):
    fallback = False
    if center is None or len(center) != 2 or not all(math.isfinite(value) for value in center):
        center = (width / 2, height / 2)
        fallback = True
    x = min(max(float(center[0]), 0.0), float(width))
    y = min(max(float(center[1]), 0.0), float(height))
    return (x, y), fallback


def generate_view(image: Image.Image, view_id: str, full_prediction_center=None) -> GeneratedView:
    if view_id not in VIEWS:
        raise ValueError(f"unknown W2 view: {view_id}")
    image = image.convert("RGB")
    width, height = image.size
    if view_id in {"full", "v4"}:
        return GeneratedView(
            image,
            ViewGeometry(view_id, image.size, image.size, 0.0, 0.0),
        )
    if view_id == "v1":
        padded = ImageOps.expand(image, border=BORDER_PIXELS, fill=(0, 0, 0))
        return GeneratedView(
            padded,
            ViewGeometry(
                view_id, image.size, padded.size,
                -float(BORDER_PIXELS), -float(BORDER_PIXELS),
            ),
        )
    center, fallback = _validated_center(full_prediction_center, width, height)
    fraction = CROP_FRACTIONS[view_id]
    crop_width = max(1, round(width * fraction))
    crop_height = max(1, round(height * fraction))
    left = round(center[0] - crop_width / 2)
    top = round(center[1] - crop_height / 2)
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return GeneratedView(
        cropped,
        ViewGeometry(
            view_id, image.size, cropped.size,
            float(left), float(top), fallback,
        ),
    )


def max_visual_tokens(bench: str, view_id: str) -> int:
    if bench == "androidcontrol":
        return 768 if view_id == "v4" else 12800
    if bench == "mind2web":
        return 768 if view_id == "v4" else 1344
    raise ValueError(f"unknown benchmark: {bench}")