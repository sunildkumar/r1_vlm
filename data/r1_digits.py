import os
import random

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from transformers import AutoProcessor

load_dotenv(dotenv_path=find_dotenv())



# Load processor and set paths
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
base_image_path = "/millcreek/home/sunil/r1_vlm/counting-mnist"


def generate_r1_messages(example, task):
    label = example["label"]
    file_path = os.path.join(base_image_path, example['image_path'])
    
    total = sum(label)
    
    if task == 'recognition':
        instruction = f"What digits are in this image? I need the digits in a list sorted from lowest to highest."
        ending = "Show your work in <think> </think> tags and return the answer in <answer> </answer> tags, for example <answer> [1, 2, 3] </answer>."
        
        instruction = f"{instruction} {ending}"

        r1_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": file_path},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me solve this step by step.\n<think>"}
                ],
            },
        ]
        
        return {
            "messages": r1_messages,
            "task": task,
            "label": label,
            "total": total,
        }
    elif task == 'addition':
        instruction = f"What is the sum of the digits in this image?"
        ending = "Show your work in <think> </think> tags and return the answer in <answer> </answer> tags, for example <answer> 3 </answer>."
 
        instruction = f"{instruction} {ending}"
        
        
        r1_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": file_path},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me solve this step by step.\n<think>"}
                ],
            },
        ]
        
        
        return {
            "messages": r1_messages,
            "task": task,
            "label": label,
            "total": total,
        }
        
    else:
        raise ValueError(f"Task {task} not supported")
        
    


def create_r1_counting_mnist_dataset():
    dataset = load_dataset("sunildkumar/digit-recognition")
    processed_datasets = {}
    for split in dataset.keys():
        print(f"Processing {split} split...")
        examples = []
        for example in tqdm(dataset[split], desc=f"Processing {split} examples"):
            for task in ['recognition', 'addition']:
                processed_example = generate_r1_messages(example, task)
                examples.append(processed_example)
                
        processed_datasets[split] = Dataset.from_list(examples)
        
    return DatasetDict(processed_datasets)

        
if __name__ == "__main__":
    dataset = create_r1_counting_mnist_dataset()
    dataset.push_to_hub(
        "sunildkumar/digit-recognition-r1", token=os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
