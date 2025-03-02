import os

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

load_dotenv(dotenv_path=find_dotenv())

# this holds the images/ directory created by message_decoding_sentences_dataset.py
base_image_path = (
    "/millcreek/home/sunil/r1_vlm/src/r1_vlm/datasets/message_decoding_sentences"
)


def generate_r1_messages(example):
    coded_message = example["coded_message"]
    decoded_message = example["decoded_message"]
    mapping = example["mapping"]
    task = example["task"]
    file_path = os.path.join(base_image_path, example["file_path"])

    assert os.path.exists(file_path), f"File does not exist: {file_path}"

    # add spaces between each character to prevent tokenization issues
    coded_message = " ".join(coded_message)

    instruction = f'Use the decoder in the image to decode this coded message: "{coded_message}". The decoded message should be one or more english words. Underscore characters ("_") in the coded message should be mapped to a space (" ") when decoding.'

    ending = "Show your work in <think> </think> tags and return the answer in <answer> </answer> tags, for example <answer> cat </answer> or <answer> this is the decoded message. </answer>. You may think for as long as you need to. You should verify your answer as carefully as possible while thinking before answering."

    instruction = f"{instruction} {ending}"

    r1_messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.",
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
        "coded_message": coded_message,
        "mapping": mapping,
        "decoded_message": decoded_message,  # No spaces here because we want the model to reply with the relevant english word(s).
        "task": task,
    }


def create_r1_message_decoding_dataset():
    dataset = load_dataset(
        "sunildkumar/message-decoding-words-and-sentences", split="train"
    )

    examples = []
    for example in tqdm(dataset, desc="Processing examples"):
        processed_example = generate_r1_messages(example)
        examples.append(processed_example)

    processed_dataset = Dataset.from_list(examples)

    splits = processed_dataset.train_test_split(test_size=0.1, seed=42)

    dataset_dict = {
        "train": splits["train"],
        "test": splits["test"],
    }

    r1_dataset = DatasetDict(dataset_dict)
    r1_dataset.push_to_hub(
        "sunildkumar/message-decoding-words-and-sentences-r1",
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )


if __name__ == "__main__":
    create_r1_message_decoding_dataset()
