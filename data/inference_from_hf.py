from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import ModelConfig
import imgcat
from PIL import Image
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

dataset = load_dataset("sunildkumar/coco-counts-r1")


for example in dataset["train"]:
    image_path = example["messages"][1]['content'][0]['image']
    class_1 = example['class_1']
    
    # choose the detection or counting message
    detection_msg = f"Detect all {class_1} in the image and return their locations in the form of coordinates."
    count_msg = f"Count the number of {class_1} in the image. To ensure accuracy, first detect their bounding boxes points and then return the count. Remember you have visual grounding capabilities. "
    
    format_msg = "The format of output boxes should be like {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"<label>\"}" 
    
    msg = count_msg + " " + format_msg
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": msg},
            ],
        }
    ]
    
    imgcat.imgcat(Image.open(image_path))
    
    

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(text)
    print(output_text)