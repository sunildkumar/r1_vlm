# vision_tool.py
from transformers import Tool
from qwen_vl_utils import QWenTokenizer, get_image_feature
import torch

class ImageLoaderTool(Tool):
    name = "image_loader"
    description = "Loads JPEG images from /data/images directory"
    inputs = {
        "filename": {"type": "str", "description": "Name of JPEG file in /data/images"}
    }
    outputs = {"image_features": {"type": "tensor", "description": "Visual features"}}

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, filename: str):
        # Load and process image
        image_path = f"/data/images/{filename}"
        image_feature = get_image_feature(image_path, self.tokenizer)
        return {"image_features": image_feature.to(self.device)}


