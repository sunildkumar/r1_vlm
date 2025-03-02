import random

from datasets import load_dataset

# creates a dataset of sentences that we will use for this task
# filters out characters that are not in the decoder

sentences_dataset = load_dataset("c2p-cmd/Good-Quotes-Authors", "HUGE", split="train")


# filter each sentence to only include characters in the decoder
def char_in_decoder(char: str) -> bool:
    return char.isascii() and (char.isalpha() or char == " ")


def clean_sentence(example: dict) -> dict:
    sentence = example["quote"]
    if sentence is None:
        return {"text": ""}
    cleaned = "".join(char for char in sentence if char_in_decoder(char))
    return {"text": cleaned}


sentences_dataset = sentences_dataset.map(clean_sentence)
sentences_dataset = sentences_dataset.remove_columns(["quote", "author", "category"])

# filter out examples that are less than 3 words long
sentences_dataset = sentences_dataset.filter(lambda x: len(x["text"].split()) >= 3)
print(
    f"After filtering for number of words per sentence, there are {len(sentences_dataset)} sentences in the dataset"
)

# filter out sentences that are too short or too long
shortest_length = 1
longest_length = 100
filtered_sentences = [
    s
    for s in sentences_dataset
    if len(s["text"]) >= shortest_length and len(s["text"]) <= longest_length
]
print(
    f"After filtering for length, there are {len(filtered_sentences)} sentences in the dataset"
)

# filter out sentences that include  underscore ("_") character (as we will use this character as a delimiter in the coded message)
filtered_sentences = [s for s in filtered_sentences if "_" not in s["text"]]
print(
    f"After filtering for underscore, there are {len(filtered_sentences)} sentences in the dataset"
)

# sort the sentences by length and interpolate over the range to get inputs of diverse lengths
dataset_size = 20000
# Group sentences into 10 length buckets (0-10, 10-20, ..., 90-100)
buckets = [[] for _ in range(10)]
for sentence in filtered_sentences:
    length = len(sentence["text"])
    bucket_index = min(length // 10, 9)  # Ensure index stays within 0-9
    buckets[bucket_index].append(sentence)

# Print distribution of sentences across buckets
for i, bucket in enumerate(buckets):
    min_len = i * 10
    max_len = (i + 1) * 10
    print(f"Length {min_len}-{max_len}: {len(bucket)} sentences")

# Select equal number from each bucket (when possible)
samples_per_bucket = dataset_size // 10
selected_sentences = []
for i, bucket in enumerate(buckets):
    min_len = i * 10
    max_len = (i + 1) * 10
    # If bucket doesn't have enough sentences, take all of them
    bucket_samples = min(samples_per_bucket, len(bucket))
    selected = random.sample(bucket, bucket_samples) if bucket_samples > 0 else []
    selected_sentences.extend(selected)
    print(f"Selected {len(selected)} sentences from length {min_len}-{max_len}")

# If we didn't reach dataset_size, fill the rest from remaining sentences
if len(selected_sentences) < dataset_size:
    remaining = dataset_size - len(selected_sentences)
    # Collect all sentences not already selected
    all_selected_ids = {id(s) for s in selected_sentences}
    remaining_sentences = [
        s for s in filtered_sentences if id(s) not in all_selected_ids
    ]
    # If we have enough remaining sentences, sample randomly
    if remaining_sentences:
        additional = random.sample(
            remaining_sentences, min(remaining, len(remaining_sentences))
        )
        selected_sentences.extend(additional)
        print(f"Added {len(additional)} additional sentences to reach target size")

# Convert back to a dataset
sentences_dataset = sentences_dataset.from_list(selected_sentences)

print(f"Selected {len(sentences_dataset)} sentences")

# upload to hub
sentences_dataset.push_to_hub("sunildkumar/english-sentences")
