# vision_tool.py
import os
import re
from pathlib import Path

from transformers import Tool
from qwen_vl_utils import QWenTokenizer, get_image_feature
import torch

class ImageLoaderTool(Tool):
    name = "image_loader"
    description = "Loads JPEG images from /data/images directory"
    inputs = {
        "filename": {"type": "str", "description": "Name of JPEG file to load"}
    }
    outputs = {"image_features": {"type": "tensor", "description": "Visual features"}}

    def __init__(self, tokenizer, data_dir: Path):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_dir = data_dir

    def forward(self, filename: str):
        try:
            self._check_filename_safety(filename)
            image_path = f"{self.data_dir}/{filename}"
            image_feature = get_image_feature(image_path, self.tokenizer)
            return {"image_features": image_feature.to(self.device)}
        except Exception as e:
            return {"error": str(e)}
    
    def _check_filename_safety(self, filename: str):
        if not filename.endswith(".jpeg"):
            raise ValueError("Filename must end with .jpeg")
        # check that the filename is nothing but alphanumeric and underscores
        if not re.match(r"^[a-zA-Z0-9_]+$", filename):
            raise ValueError("Filename must contain only alphanumeric characters and underscores")

        if not os.path.exists(f"{self.data_dir}/{filename}"):
            raise ValueError("File does not exist")


