from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import ModelConfig

dataset = load_dataset("sunildkumar/coco-computation-r1")

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", padding_side="left"
)

dataset = load_dataset("sunildkumar/coco-computation-r1")


example = dataset["train"][0]["messages"]
for message in example:
    content = message["content"]
    message["content"] = [
        {k: v for k, v in item.items() if v is not None} for item in content
    ]


texts = processor.apply_chat_template(
    example, continue_final_message=True, tokenize=False
)

print(texts)

image_input, _ = process_vision_info(example)

batch = processor(
    text=texts,
    images=image_input,
    padding=True,
    return_tensors="pt",
)

batch = batch.to("cuda")

# load the model
model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="bfloat16",
    use_peft=True,
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_config.model_name_or_path,
    torch_dtype=model_config.torch_dtype,
    # has to be set to false for gradient checkpointing to work
    use_cache=False,
    device_map="auto",
)


output = model.generate(**batch, max_length=100000)

print(processor.decode(output[0], skip_special_tokens=False))
