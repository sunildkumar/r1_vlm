import os
import random
from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm

from r1_vlm.datasets.message_decoding_words.message_decoding_words_dataset import (
    generate_decoder_image,
    generate_mapping,
)

# setting a seed for reproducibility
random.seed(42)


def create_sample(example):
    sentence = example["text"] if "text" in example else example["word"]

    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    assert len(alphabet) == 26
    mapping = generate_mapping(alphabet)

    decoder_image = generate_decoder_image(mapping)

    # add a mapping for the underscore ("_") character. It will map to " " (space).
    # This is so we can effectively communicate the space character in the coded message.
    mapping["_"] = " "

    # reverse the mapping to encode the sentence
    reverse_mapping = {v: k for k, v in mapping.items()}

    # create the coded and decoded sentence. If we encounter a character that is not in the mapping,
    # we will map it to itself.
    coded_sentence = ""
    decoded_sentence = ""
    for char in sentence:
        # check if the character is in the mapping
        if char.isascii() and (char.isalpha() or char == " "):
            is_lower = char.islower()

            # pass the lowercase version of the character to the mapping
            # .lower() on the space character is a no-op
            key_char = char.lower() if not is_lower else char
            mapped_char = reverse_mapping[key_char]

            # add lowercase char to the coded sentence
            coded_sentence += mapped_char

            # use the lowercase version of the character in the decoded sentence too.
            decoded_sentence += char.lower() if not is_lower else char
        # if the character is not in the mapping, we will map it to itself
        else:
            coded_sentence += char
            decoded_sentence += char

    return decoder_image, decoded_sentence, coded_sentence, mapping


def create_dataset():
    data = {
        "coded_message": [],
        "decoded_message": [],
        "mapping": [],
        "file_path": [],
        "image": [],
        "task": [],
    }

    image_dir = Path(__file__).parent / "images"
    image_dir.mkdir(exist_ok=True)

    # verify that the image directory is empty
    if len(list(image_dir.glob("*.png"))) > 0:
        raise ValueError("Image directory is not empty")

    words_dataset = load_dataset("sunildkumar/popular_english_words", split="train")
    for i, example in tqdm(enumerate(words_dataset), total=len(words_dataset)):
        word = example["word"]

        decoder_image, sentence, coded_sentence, mapping = create_sample({"word": word})

        fpath = f"images/{i}.png"
        full_path = image_dir / f"{i}.png"

        # verify that the image doesn't already exist, if it does, something is wrong as we should error
        if full_path.exists():
            raise ValueError(f"Image {full_path} already exists")

        # Use full path for saving the image
        full_path = image_dir / f"{i}.png"

        decoder_image.save(full_path)

        data["coded_message"].append(coded_sentence)
        data["decoded_message"].append(sentence)
        data["mapping"].append(mapping)
        data["file_path"].append(fpath)
        data["image"].append(decoder_image)
        data["task"].append("word")

    # Get the index we should start numbering from for the words dataset
    last_index = len(words_dataset)

    sentences_dataset = load_dataset("sunildkumar/english-sentences", split="train")

    for i, example in tqdm(enumerate(sentences_dataset), total=len(sentences_dataset)):
        decoder_image, sentence, coded_sentence, mapping = create_sample(example)

        current_index = last_index + i

        # Store only the relative path starting from 'images/'
        fpath = f"images/{current_index}.png"

        # Use full path for saving the image
        full_path = image_dir / f"{current_index}.png"

        # verify that the image doesn't already exist, if it does, something is wrong as we should error
        if full_path.exists():
            raise ValueError(f"Image {full_path} already exists")

        decoder_image.save(full_path)

        data["coded_message"].append(coded_sentence)
        data["decoded_message"].append(sentence)
        data["mapping"].append(mapping)
        data["file_path"].append(fpath)
        data["image"].append(decoder_image)
        data["task"].append("sentence")

    # Get the index we should start numbering from for the word pairs
    last_index = last_index + len(sentences_dataset)

    # Create word pairs task
    words_list = [example["word"] for example in words_dataset]
    num_pairs = len(words_dataset) // 2  # Create half as many pairs as there are words

    for i in tqdm(range(num_pairs), total=num_pairs):
        # Select two random words
        word1, word2 = random.sample(words_list, 2)

        # Combine words
        word_pair = f"{word1} {word2}"

        decoder_image, sentence, coded_sentence, mapping = create_sample(
            {"word": word_pair}
        )

        current_index = last_index + i
        fpath = f"images/{current_index}.png"
        full_path = image_dir / f"{current_index}.png"

        if full_path.exists():
            raise ValueError(f"Image {full_path} already exists")

        decoder_image.save(full_path)

        data["coded_message"].append(coded_sentence)
        data["decoded_message"].append(sentence)
        data["mapping"].append(mapping)
        data["file_path"].append(fpath)
        data["image"].append(decoder_image)
        data["task"].append("word_pair")

    decoding_dataset = Dataset.from_dict(data)

    decoding_dataset.push_to_hub(
        "sunildkumar/message-decoding-words-and-sentences",
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )


if __name__ == "__main__":
    create_dataset()
