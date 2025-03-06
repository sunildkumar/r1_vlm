# seeing if we can visualize the attention weights during decoding
# run with CUDA_VISIBLE_DEVICES=0,1 uv run attention_demo/demo.py
import imageio.v3 as imageio
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def visualize_attention_step(
    attention_weights_step,
    token_ids,
    processor,
    start_phrase="The coded message is",
    use_color=False,
):
    # attention_weights_step shape is [1, 16, seq_len, seq_len]
    # First squeeze out the batch dimension
    attention = attention_weights_step.squeeze(0)  # now [16, seq_len, seq_len]

    # Get attention weights for the last generated token (last position)
    # Average across attention heads
    last_token_attention = attention[:, -1, :].mean(dim=0).detach().cpu().float()

    # renormalize the attention weights so they sum to 1
    last_token_attention = last_token_attention / last_token_attention.sum()
    attention_weights_np = last_token_attention.numpy()

    # Apply non-linear scaling to enhance visibility of medium attention weights
    # First normalize to [0,1] range
    min_val = attention_weights_np.min()
    max_val = attention_weights_np.max()
    if max_val > min_val:
        normalized_weights = (attention_weights_np - min_val) / (max_val - min_val)
        # Apply power scaling (values less than 1 will be boosted)
        scaled_weights = np.power(
            normalized_weights, 0.4
        )  # Adjust power value to control contrast
    else:
        scaled_weights = attention_weights_np

    # Decode all tokens first
    tokens = processor.batch_decode(
        token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    # Find the start index by looking for the start phrase
    full_text = "".join(tokens)
    start_idx = full_text.find(start_phrase)

    # Convert character index to token index by counting tokens up to that point
    token_start_idx = 0
    char_count = 0
    for i, token in enumerate(tokens):
        char_count += len(token)
        if char_count > start_idx:
            token_start_idx = i
            break

    # Truncate tokens and weights to start from the found position
    tokens = tokens[token_start_idx:]
    scaled_weights = scaled_weights[token_start_idx:]

    # Create a black image (changed from white to black)
    img_width = 2500
    img_height = 2500
    image = Image.new("RGB", (img_width, img_height), "black")
    draw = ImageDraw.Draw(image)

    # Try to load a monospace font, fallback to default if not available
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 50
        )
    except:
        font = ImageFont.load_default()

    # Calculate text positions
    x, y = 40, 40
    max_width = img_width - 80
    line_height = 40

    for i, (token, weight) in enumerate(zip(tokens, scaled_weights)):
        # Handle newlines in token - check for both \n and {data}\n patterns
        if "\n" in token or "\\n" in token:
            # Split token into parts before and after newline
            parts = token.replace("\\n", "\n").split("\n")

            for j, part in enumerate(parts):
                if part:  # If there's content, render it
                    bbox = draw.textbbox((x, y), part, font=font)
                    text_width = bbox[2] - bbox[0]

                    # Check if we need to start a new line
                    if x + text_width > max_width:
                        x = 40
                        y += line_height

                    # For the last token (current generation), use blue
                    if i == len(tokens) - 1:
                        color = (100, 150, 255)  # Light blue for current token
                    else:
                        if use_color:
                            # Create red to green gradient based on attention weight
                            red = int(255 * (1 - weight))
                            green = int(255 * weight)
                            color = (red, green, 0)
                        else:
                            color = (
                                255,
                                255,
                                255,
                            )  # White color when use_color is False

                    # Draw the part
                    draw.text((x, y), part, fill=color, font=font)
                    x += text_width + 4

                # Move to next line if there are more parts or if this isn't the last empty part
                if j < len(parts) - 1 or (j == len(parts) - 1 and not part):
                    x = 40
                    y += line_height

            continue  # Skip the rest of the loop for this token

        # Normal token handling (no newline)
        bbox = draw.textbbox((x, y), token, font=font)
        text_width = bbox[2] - bbox[0]

        # Check if we need to start a new line
        if x + text_width > max_width:
            x = 40  # Reset x to start of line
            y += line_height  # Move to next line

        # For the last token (current generation), use blue
        if i == len(tokens) - 1:
            color = (100, 150, 255)  # Light blue for current token
        else:
            if use_color:
                # Create red to green gradient based on attention weight
                red = int(255 * (1 - weight))
                green = int(255 * weight)
                color = (red, green, 0)
            else:
                color = (
                    255,
                    255,
                    255,
                )  # White color when use_color is False (changed from black)

        # Draw the token
        draw.text((x, y), token, fill=color, font=font)

        # Move x position for next token (add small space between tokens)
        x += text_width + 4

    # Convert PIL image to numpy array
    return np.array(image)


def combine_attention_videos(text_frames, image_frames):
    """
    Combines text attention and image attention frames side by side.

    Args:
        text_frames: List of numpy arrays containing text attention visualization frames
        image_frames: List of numpy arrays containing image attention visualization frames

    Returns:
        List of numpy arrays containing combined frames
    """
    assert len(text_frames) == len(image_frames), "Number of frames must match"

    combined_frames = []
    for text_frame, image_frame in zip(text_frames, image_frames):
        # Get dimensions
        text_height, text_width = text_frame.shape[:2]
        image_padding_width = text_width // 2  # Space to allocate for image

        # extend the right edge of the text frame
        extended_width = text_width + image_padding_width
        canvas = np.zeros(
            (text_height, extended_width, 3), dtype=np.uint8
        )  # Black background

        # Copy the text frame to the left side
        canvas[:, :text_width] = text_frame

        # resize image
        image_width = int(0.55 * text_height)
        image_height = image_width

        image_frame_resized = Image.fromarray(image_frame).resize(
            (image_width, image_height), Image.Resampling.LANCZOS
        )
        image_frame_resized = np.array(image_frame_resized)

        # place image on the canvas - define the top left corner
        # 1450 works
        top_left_x = 1000
        top_left_y = 600

        combined_frame = canvas.copy()

        combined_frame[
            top_left_y : top_left_y + image_height,
            top_left_x : top_left_x + image_width,
        ] = image_frame_resized  # noqa: E203

        # crop to where the image ends in x plus padding
        top_right_x = top_left_x + image_width + 20

        # Ensure the width is even (required by video codecs)
        if top_right_x % 2 != 0:
            top_right_x += 1  # Make it even by adding 1 if it's odd

        combined_frame = combined_frame[:, :top_right_x]  # noqa: E203

        combined_frames.append(combined_frame)

    return combined_frames


def create_attention_visualization(
    attention_weights,
    sequences,
    processor,
    layer_idx=-1,
    fps=2,
    output_path="attention_visualization.mp4",
    start_phrase="The coded message is",
    use_color=False,
):
    """
    Create a video visualization of attention weights during generation.

    Args:
        attention_weights: List of attention weights from model generation
        sequences: Token sequences from model generation
        processor: Tokenizer/processor for decoding tokens
        layer_idx: Index of attention layer to visualize (default: -1 for last layer)
        fps: Frames per second for output video (default: 2)
        output_path: Path to save the output video (default: "attention_visualization.mp4")
        start_phrase: Phrase to start visualization from (default: "The coded message is")
        use_color: Whether to color the text according to attention weights (default: True)
    """
    num_steps = len(attention_weights)
    base_sequence = sequences.shape[1] - num_steps

    # Store frames in memory
    frames = []

    for step in tqdm(range(1, num_steps)):  # start from 1 as step 0 is just the input
        attention_weights_step = attention_weights[step][
            layer_idx
        ]  # get specified layer's attention
        current_tokens = sequences[0][: base_sequence + step]
        frame = visualize_attention_step(
            attention_weights_step,
            current_tokens,
            processor,
            start_phrase=start_phrase,
            use_color=use_color,
        )
        frames.append(frame)

    return frames


def visualize_image_attention(
    inputs,
    image,
    attention_weights,
    sequences,
    processor,
):
    # get the patch grid
    _, h, w = inputs["image_grid_thw"].cpu().numpy().squeeze(0)

    # handle patch merging
    merge_size = processor.image_processor.merge_size
    h = h // merge_size
    w = w // merge_size

    total_patches = h * w

    # there should be this many image tokens in the input
    image_pad_token = "<|image_pad|>"
    image_pad_id = processor.tokenizer.convert_tokens_to_ids(image_pad_token)

    num_image_tokens = (inputs["input_ids"] == image_pad_id).sum().cpu().numpy().item()

    assert num_image_tokens == total_patches, (
        f"Expected {num_image_tokens=} to equal {total_patches=}"
    )

    # attention_weights shape is [1, 16, seq_len, seq_len]
    # First squeeze out the batch dimension
    attention = attention_weights.squeeze(0)  # now [16, seq_len, seq_len]

    # Get attention weights for the last generated token (last position)
    # Average across attention heads
    last_token_attention = attention[:, -1, :].mean(dim=0).detach().cpu().float()

    # renormalize the attention weights so they sum to 1
    last_token_attention = last_token_attention / last_token_attention.sum()
    attention_weights_np = last_token_attention.numpy()

    # now we should select the attention weights corresponding to the image tokens
    image_tokens_mask = (inputs["input_ids"] == image_pad_id).cpu().numpy().squeeze(0)
    # pad the mask on the right with False's - these are generated tokens
    image_tokens_mask = np.pad(
        image_tokens_mask,
        (0, attention_weights_np.shape[0] - image_tokens_mask.shape[0]),
        mode="constant",
        constant_values=False,
    )

    # these should be the same shape before we apply the mask
    assert image_tokens_mask.shape == attention_weights_np.shape, (
        f"The image tokens mask and attention weights shape mismatch: {image_tokens_mask.shape=} {attention_weights_np.shape=}"
    )

    # now we should select the attention weights corresponding to the image tokens
    attention_weights_np = attention_weights_np[image_tokens_mask]

    # we should have one attention weight per image token
    assert num_image_tokens == attention_weights_np.shape[0], (
        f"Expected {num_image_tokens=} to equal {attention_weights_np.shape[0]=}, as there should be one attention weight per image token"
    )

    # Create a transparent overlay for the grid
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Calculate the size of each grid cell in pixels
    width, height = image.size
    cell_width = width // w
    cell_height = height // h

    # Draw horizontal lines (black with 50% transparency)
    for i in range(h + 1):
        y = i * cell_height
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, 128), width=1)

    # Draw vertical lines (black with 50% transparency)
    for j in range(w + 1):
        x = j * cell_width
        draw.line([(x, 0), (x, height)], fill=(0, 0, 0, 128), width=1)

    # Apply non-linear scaling to enhance visibility of medium attention weights
    min_val = attention_weights_np.min()
    max_val = attention_weights_np.max()
    if max_val > min_val:
        normalized_weights = (attention_weights_np - min_val) / (max_val - min_val)
        # Apply power scaling (values less than 1 will be boosted)
        scaled_weights = np.power(normalized_weights, 0.4)
    else:
        scaled_weights = attention_weights_np

    # Fill each grid cell with attention-based color
    for idx, weight in enumerate(scaled_weights):
        # Calculate grid position
        grid_x = idx % w
        grid_y = idx // w

        # Calculate pixel coordinates
        x1 = grid_x * cell_width
        y1 = grid_y * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height

        # Create red to green gradient based on attention weight
        red = int(255 * (1 - weight))
        green = int(255 * weight)
        # Add semi-transparent color overlay
        draw.rectangle([x1, y1, x2, y2], fill=(red, green, 0, 128))

    # Combine the original image with the overlay
    image = image.convert("RGBA")
    grid_image = Image.alpha_composite(image, overlay)

    # convert back into RGB
    grid_image = grid_image.convert("RGB")

    grid_image = np.array(grid_image)

    return grid_image


def create_image_attention_demo(
    inputs,
    image,
    attention_weights,
    sequences,
    processor,
    layer_idx=-1,
    fps=2,
    output_path="visual_attention_demo.mp4",
):
    """
    Args:
        inputs: Inputs to the model
        image: PIL image that was passed to the model
        attention_weights: Attention weights from the model during generation
        sequences: Generated sequences from the model during generation
    """
    num_steps = len(attention_weights)
    base_sequence = sequences.shape[1] - num_steps

    # Store frames in memory
    frames = []

    for step in tqdm(range(1, num_steps)):  # start from 1 as step 0 is just the input
        attention_weights_step = attention_weights[step][
            layer_idx
        ]  # get specified layer's attention
        current_tokens = sequences[0][: base_sequence + step]

        frame = visualize_image_attention(
            inputs, image, attention_weights_step, current_tokens, processor
        )
        frames.append(frame)

    return frames


if __name__ == "__main__":
    checkpoint = "/millcreek/home/sunil/r1_vlm/vlm-r1-message-decoding-words-and-sequences_official_demo/checkpoint-1850"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=checkpoint,
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print("model loaded")

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=checkpoint,
        padding_side="left",
    )

    dataset = load_dataset("sunildkumar/message-decoding-words-and-sequences-r1")[
        "train"
    ]

    # choose a random element of the dataset
    example = dataset[np.random.randint(0, len(dataset))]

    # inject our message in place of the original one
    message = "im trained with grpo"

    # how to encode message - we need to reverse the mapping
    mapping = example["mapping"]
    reverse_mapping = {v: k for k, v in mapping.items()}

    # add space
    reverse_mapping[" "] = "_"

    # encode the message
    encoded_message = [reverse_mapping[char] for char in message]

    # space it out
    encoded_message = " ".join(encoded_message)

    # inject it into the original example
    instruction_text = example["messages"][1]["content"][-1]["text"]
    base_text = instruction_text[: instruction_text.rindex(":") + 1]
    example["messages"][1]["content"][-1]["text"] = f"{base_text} {encoded_message}."

    example["decoded_message"] = message
    example["coded_message"] = encoded_message

    messages = example["messages"]
    for message in messages:
        content = message["content"]
        message["content"] = [
            {k: v for k, v in item.items() if v is not None} for item in content
        ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    print("Starting generation")
    with torch.no_grad():
        generated_output = model.generate(
            **inputs,
            temperature=1.0,
            max_new_tokens=512,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    print("Generation complete")

    generated_text = processor.decode(
        generated_output.sequences[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    # print the generated text
    print(generated_text)
    import ipdb

    ipdb.set_trace()

    # create visualizations for layer 20
    layer_idx = 20

    # Get frames for both visualizations
    text_frames = create_attention_visualization(
        generated_output.attentions,
        generated_output.sequences,
        processor,
        layer_idx=layer_idx,
    )

    image_frames = create_image_attention_demo(
        inputs,
        image_inputs[0],
        generated_output.attentions,
        generated_output.sequences,
        processor,
        layer_idx=layer_idx,
    )

    # Combine frames and save video
    combined_frames = combine_attention_videos(text_frames, image_frames)
    # Save videos at different frame rates
    for fps in [2, 5, 10]:
        output_path = f"combined_attention_visualization_layer{layer_idx}_{fps}fps.mp4"
        print(f"Saving video at {fps} fps to {output_path}")
        imageio.imwrite(output_path, combined_frames, fps=fps, codec="libx264")
