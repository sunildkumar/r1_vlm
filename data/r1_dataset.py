import os

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from transformers import AutoProcessor

load_dotenv(dotenv_path=find_dotenv())


# Load processor and set paths
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
base_image_path = "/millcreek/data/academic/coco"


# gemerate r1 prompt with a prefix for the model to already start with the thinking process
def generate_r1_messages(example):
    split = example["split"]
    image_path = os.path.join(base_image_path, f"{split}2017", example["file_name"])

    # choose the appropriate message based on the operation
    operation = example["operation"]
    if operation == "add":
        operation_message = f"For this image, add the number of {example['class_1']} to the number of {example['class_2']}. "
    elif operation == "subtract":
        operation_message = f"For this image, subtract the number of {example['class_2']} from the number of {example['class_1']}. "
    elif operation == "multiply":
        operation_message = f"For this image, multiply the number of {example['class_1']} by the number of {example['class_2']}. "
    elif operation == "divide":
        operation_message = f"For this image, divide the number of {example['class_1']} by the number of {example['class_2']}. "

    # add standard ending and grounding reminder to the message
    ending = "Show your work in <think> </think> tags and return the answer in <answer> </answer> tags. If the answer is not an integer, truncate it to 2 decimal places. "
    grounding = 'Remember you have visual grounding capabilities and you can output bbox coordinates or key points in JSON format. Bbox example: {"bbox_2d": [74, 58, 526, 619], "label": "person"}. Keypoint example: {"point_2d": ["38", "314"], "label": "person"}. You should NOT attempt to count without using visual grounding as it is not accurate.'
    operation_message += ending + grounding

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
                {"type": "text", "text": operation_message},
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
        "target": example["answer"],
        "class_1": example["class_1"],
        "class_2": example["class_2"],
        "count_1": example["count_1"],
        "count_2": example["count_2"],
    }


def create_r1_dataset():
    # Load source dataset with all splits
    dataset = load_dataset("sunildkumar/coco-computation")

    processed_datasets = {}
    for split in dataset.keys():
        print(f"Processing {split} split...")
        examples = []
        for example in tqdm(dataset[split], desc=f"Processing {split} examples"):
            processed_example = generate_r1_messages(example)
            examples.append(processed_example)

        processed_datasets[split] = Dataset.from_list(examples)

    return DatasetDict(processed_datasets)


if __name__ == "__main__":
    # Create dataset with all splits
    final_dataset = create_r1_dataset()

    # Push to Hugging Face Hub
    final_dataset.push_to_hub(
        "sunildkumar/coco-computation-r1", token=os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
