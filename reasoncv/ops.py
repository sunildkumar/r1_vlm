from dataclasses import dataclass
import logging
from typing import ClassVar, List

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from reasoncv.opbase import RcvOpRequest, ConversationContext

logger = logging.getLogger(__name__)


def available_ops() -> List[str]:
    """Returns a sorted list of available operation names"""
    return sorted(RcvOpRequest.registry.keys())


def xpixels(img: Image.Image, val: float) -> int:
    """
    Returns the x-coordinate in pixels for a given normalized value.
    """
    width = img.size[0]
    return int(val * width)


def ypixels(img: Image.Image, val: float) -> int:
    """
    Returns the y-coordinate in pixels for a given normalized value.
    """
    height = img.size[1]
    return int(val * height)


class ZoomOp(RcvOpRequest):
    # using ClassVar to avoid Pydantic treating this as a field
    _name: ClassVar[str] = "zoom"  
    center: tuple[float, float]
    size: tuple[float, float]

    def bbox(self, image: Image.Image) -> tuple[int, int, int, int]:
        """
        Returns the bounding box of the zoom operation, in pixels
        """
        cx, cy = self.center
        w, h = self.size
        x0 = xpixels(image, cx - w / 2)
        y0 = ypixels(image, cy - h / 2)
        x1 = xpixels(image, cx + w / 2)
        y1 = ypixels(image, cy + h / 2)
        return x0, y0, x1, y1

    def run(self, image: Image.Image, context: ConversationContext):
        """
        """
        x0, y0, x1, y1 = self.bbox(image)
        cropped = image.crop((x0, y0, x1, y1))
        context.add_image(cropped)


@dataclass
class BboxResult:
    bbox: tuple[float, float, float, float]  # Normalized 0-1
    score: float  # Normalized 0-1

    def __str__(self):
        x0, y0, x1, y1 = self.bbox
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        width = x1 - x0
        height = y1 - y0
        return f"Bbox:\n  center: [{center_x:.3f}, {center_y:.3f}]\n  size: [{width:.3f}, {height:.3f}]\n  score: {self.score:.3f}"

class ObjectDetector():
    """Base class for object detectors."""

    def detect_objects(self, image: Image.Image, class_name: str) -> list[BboxResult]:
        raise NotImplementedError("Subclasses must implement this method")


class GroundingDINODetector(ObjectDetector):
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def detect_objects(self, image: Image.Image, class_name: str, threshold: float = 0.5) -> list[BboxResult]:
        inputs = self.processor(text=class_name, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
        
        bounding_boxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            bbox_xyxy = box.tolist()
            # Normalize the xyxy to 0-1
            x0, y0, x1, y1 = bbox_xyxy
            h, w = image.height, image.width
            bbox_res = BboxResult(bbox=(x0/w, y0/h, x1/w, y1/h), score=score.item())
            bounding_boxes.append(bbox_res)
        
        bounding_boxes.sort(key=lambda x: x.score, reverse=True)
        return bounding_boxes



class ObjectDetectionOp(RcvOpRequest):
    """Runs Grounding DINO model on the image, and adds the bounding boxes to the context.
    """
    _name: ClassVar[str] = "object-detection"
    object_name: str
    threshold: float

    @classmethod
    def model(cls, variant: str = "tiny") -> tuple[nn.Module, object]:
        """Returns the model and processor for object detection."""
        model_id = f"IDEA-Research/grounding-dino-{variant}"

        if not hasattr(cls, "_model"):
            cls._model = GroundingDINODetector(model_id=model_id)
        return cls._model

    def bbox_to_text(self, bounding_boxes: List[BboxResult]) -> str:
        """Converts the bounding boxes to a text string."""
        n = len(bounding_boxes)
        out = f"Found {n} objects.\n"
        for i, bbox in enumerate(bounding_boxes):
            out += f"{bbox}\n"
        return out

    def run(self, image: Image.Image, context: ConversationContext):
        """Runs Grounding DINO model on the image, and adds the bounding boxes to the context."""
        model = self.model()
        bounding_boxes = model.detect_objects(image, self.object_name, self.threshold)
        bbox_text = self.bbox_to_text(bounding_boxes)
        context.add_text(bbox_text)


class DepthEstimationOp(RcvOpRequest):
    """Runs MiDaS depth estimation model on the image and adds the depth map visualization to the context.
    """
    _name: ClassVar[str] = "depth-estimation"

    @classmethod
    def model(cls) -> tuple[nn.Module, object]:
        """Returns the MiDaS model and transform pipeline for depth estimation."""
        from torchvision.transforms import Compose, Resize, ToTensor, Normalize

        if not hasattr(cls, "_model"):
            cls._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            cls._model.eval()
            if torch.cuda.is_available():
                cls._model.to("cuda")

            # MiDaS transform pipeline
            cls._transforms = Compose([
                Resize((256, 384)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return cls._model, cls._transforms

    def run(self, image: Image.Image, context: ConversationContext):
        """Runs MiDaS depth estimation on the image and adds visualization to the context.
        """
        model, transforms = self.model()
        
        # Transform and add batch dimension
        input_batch = transforms(image).unsqueeze(0)
        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")

        with torch.no_grad():
            depth_output = model(input_batch)
            if isinstance(depth_output, tuple):
                depth_output = depth_output[0]

        depth_map = depth_output.squeeze().cpu().numpy()

        context.add_text(
            f"Depth map statistics:\n"
            f"Shape: {depth_map.shape}\n"
            f"Min depth: {depth_map.min():.2f}\n"
            f"Max depth: {depth_map.max():.2f}"
        )
        
        # Normalize to 0-255 for visualization
        depth_vis = (
            ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255)
            .astype("uint8")
        )
        depth_image = Image.fromarray(depth_vis)
        context.add_image(depth_image)

