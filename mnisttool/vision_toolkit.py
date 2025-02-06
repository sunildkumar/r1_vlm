# vision_toolkit.py
from transformers import Tool, ProcessorMixin
from qwen_vl_utils import QWenTokenizer, get_image_feature
import torch

class MultiModalProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    
    def __init__(self, tokenizer, tools):
        super().__init__()
        self.tokenizer = tokenizer
        self.tools = {tool.name: tool for tool in tools}
        
    def __call__(self, examples):
        # Process text inputs
        text_inputs = self.tokenizer(
            examples["prompt"], 
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Process tool outputs
        tool_outputs = []
        for ex in examples:
            tool_name = ex.get("tool_name")
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                output = tool(ex["tool_args"])
                tool_outputs.append(output)
        
        return {
            **text_inputs,
            "tool_outputs": tool_outputs,
            "return_tensors": "pt"
        }

class ImageLoaderTool(Tool):
    name = "image_loader"
    description = "Loads JPEG images from /data/images directory"
    
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def forward(self, filename: str):
        image_path = f"/data/images/{filename}"
        return get_image_feature(image_path).to(self.device)

class ToolResponseTokenizer:
    def __init__(self, tokenizer, tools):
        self.tokenizer = tokenizer
        self.tool_processor = MultiModalProcessor(tokenizer, tools)
        
    def __call__(self, examples):
        processed = self.tool_processor(examples)
        
        # Convert tool outputs to model's expected format
        image_features = torch.stack(
            [out["image_features"] for out in processed["tool_outputs"]]
        ) if "image_loader" in examples else None
        
        return {
            "input_ids": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
            "pixel_values": image_features
        }

