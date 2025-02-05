import os
import random

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from transformers import AutoProcessor

load_dotenv(dotenv_path=find_dotenv())


# Load processor and set paths
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
base_image_path = "/millcreek/data/academic/coco"


def generate_r1_messages(example, split):
    image_path = os.path.join(base_image_path, f"{split}2017", example["file_name"])

    # Process class counts to create message
    present_objects = {
        k: v for k, v in example["class_counts"].items() if v is not None
    }
    if not present_objects:
        return None

    # choose a class at random from the present objects
    class_1 = random.choice(list(present_objects.keys()))

    count_1 = present_objects[class_1]

    counting_message = f"Count how many {class_1}(s) there are in the image."
    grounding_reminder = 'Remember you have visual grounding capabilities and you can output bbox coordinates or key points in JSON format. Bbox format: {"bbox_2d": [74, 58, 526, 619], "label": "person"}. Keypoint format: {"point_2d": ["38", "314"], "label": "person"}. You should NOT attempt to count without using visual grounding as it is not accurate.'
    ending = "Show your work in <think> </think> tags and return the answer in <answer> </answer> tags."

    instruction = f"{counting_message} {grounding_reminder} {ending}"

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
                {"type": "image", "image": image_path},
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
        "target": count_1,
        "class_1": class_1,
    }


def create_r1_counting_dataset():
    # load source dataset with all splits
    dataset = load_dataset("sunildkumar/coco-class-counts")

    processed_datasets = {}
    for split in dataset.keys():
        print(f"Processing {split} split...")
        examples = []
        for example in tqdm(dataset[split], desc=f"Processing {split} examples"):
            processed_example = generate_r1_messages(example, split)
            if processed_example:
                examples.append(processed_example)

        processed_datasets[split] = Dataset.from_list(examples)

    return DatasetDict(processed_datasets)


if __name__ == "__main__":
    dataset = create_r1_counting_dataset()
    dataset.push_to_hub(
        "sunildkumar/coco-counts-r1", token=os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
