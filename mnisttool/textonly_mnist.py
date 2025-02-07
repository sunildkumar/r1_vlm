"""A HF dataset / task for warming up with Visual Reasoning.
This dataset is based on the obfuscated MNIST data (in makedata.py) but doesn't use images at all.
The task is: Given the filename of an obfuscated MNIST image, output the correct label.
The tool available is one which can convert the obfuscated MNIST filename into the correct label, as text.
So this just checks that it can use the tool to get the data it needs to answer the task.
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


class TextOnlyMNIST:

    def __init__(self, mnist_codes_path: str | None = None):
        if mnist_codes_path:
            self.base_image_path = mnist_codes_path
        else:
            self.base_image_path = self.get_base_image_path()
        self.files = self._list_files()
        self.codes = list(self.files.keys())
        random.shuffle(self.codes)

    def get_base_image_path(self):
        home = os.path.expanduser("~")
        return os.path.join(home, "data", "obfuscated_mnist")

    def image_path(self, code: str) -> str:
        return os.path.join(self.base_image_path, f"img-{code}.jpeg")

    def random_image_path(self) -> str:
        random_code = random.choice(self.codes)
        return self.image_path(random_code)

    def generate_r1_messages(self, code:str) -> dict:
        counting_message = f"Use the tool to load the the image named 'img-{code}.jpeg' and tell me what number it is."
        ending = "Show your work in <think> </think> tags and return the answer in <answer> </answer> tags."
        tool_message = "You can invoke a tool by thinking <op>load: filename</op>"

        instruction = f"{counting_message} {ending}"

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer. {tool_message}",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.random_image_path()},
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
            "messages": messages,
            "target": float(self.files[code]["label"]),
        }


    def _list_files(self) -> dict[str, dict]:
        """Scans the files on the disk in the base_image_path
        """
        files = {}
        for file in os.listdir(self.base_image_path):
            if file.endswith(".json"):
                # Confirm filename looks like "img-abcde.json" and capture the abcde
                m = re.match(r"^img-(\w+)\.json$", file)
                if not m:
                    continue
                code = m.group(1)
                full_path = os.path.join(self.base_image_path, file)
                data = json.load(open(full_path))
                files[code] = data
        return files


    def generate_split(self, split: str) -> Dataset:
        num_codes = len(self.codes)
        if split == "train":
            codes = self.codes[:int(num_codes * 0.8)]
        elif split == "validation":
            codes = self.codes[int(num_codes * 0.8):]
        else:
            raise ValueError(f"Invalid split: {split}")

        msgs = []
        for code in tqdm(codes, desc=f"{split} messages"):
            msgs.append(self.generate_r1_messages(code))
        return Dataset.from_list(msgs)


def main(args: argparse.Namespace):
    thing = TextOnlyMNIST(args.mnist_codes_path)
    all = {}
    for split in ["train", "validation"]:
        dataset = thing.generate_split(split)
        all[split] = dataset
    dsd = DatasetDict(all)
    save_to = Path(args.save_path).expanduser()
    dsd.save_to_disk(save_to)
    print(f"Saved to {save_to}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--save_path", type=str, required=True)
    args.add_argument("--mnist_codes_path", type=str, required=False)
    args = args.parse_args()
    main(args)
