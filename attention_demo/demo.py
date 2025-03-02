# seeing if we can visualize the attention weights during decoding
# run with CUDA_VISIBLE_DEVICES=0 uv run attention_demo/demo.py
import imageio.v3 as imageio
import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def visualize_attention_step(attention_weights_step, token_ids, processor):
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

    tokens = processor.batch_decode(
        token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    # Create a white image
    img_width = 1200
    img_height = 800
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    # Try to load a monospace font, fallback to default if not available
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14
        )
    except:
        font = ImageFont.load_default()

    # Calculate text positions
    x, y = 20, 20  # Starting position
    max_width = img_width - 40  # Margin on both sides
    line_height = 20

    for i, (token, weight) in enumerate(zip(tokens, scaled_weights)):
        # Handle newlines in token
        if "\\n" in token:
            x = 20  # Reset x to start of line
            y += line_height  # Move to next line
            token = token.replace("\\n", "")  # Remove the newline for display
            if not token:  # Skip empty tokens after removing newline
                continue

        # Get text size for this token
        bbox = draw.textbbox((x, y), token, font=font)
        text_width = bbox[2] - bbox[0]

        # Check if we need to start a new line
        if x + text_width > max_width:
            x = 20  # Reset x to start of line
            y += line_height  # Move to next line

        # For the last token (current generation), use blue
        if i == len(tokens) - 1:
            color = (0, 0, 255)  # Blue for current token
        else:
            # Create red to green gradient based on attention weight
            red = int(255 * (1 - weight))
            green = int(255 * weight)
            color = (red, green, 0)

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
        # Calculate target height (same as text frame)
        target_height = text_frame.shape[0]
        target_width = text_frame.shape[1] // 2  # Half the width of text frame

        # Calculate scaling factor to maintain aspect ratio
        image_aspect = image_frame.shape[1] / image_frame.shape[0]
        target_aspect = target_width / target_height

        if image_aspect > target_aspect:
            # Image is wider than target - fit to width
            new_width = target_width
            new_height = int(target_width / image_aspect)
            vertical_padding = (target_height - new_height) // 2
            horizontal_padding = 0
        else:
            # Image is taller than target - fit to height
            new_height = target_height
            new_width = int(target_height * image_aspect)
            horizontal_padding = (target_width - new_width) // 2
            vertical_padding = 0

        # Resize image maintaining aspect ratio
        image_frame_resized = Image.fromarray(image_frame).resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )
        image_frame_resized = np.array(image_frame_resized)

        # Create a white canvas of target size
        canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)

        # Place resized image in center of canvas
        y_start = vertical_padding
        y_end = y_start + new_height
        x_start = horizontal_padding
        x_end = x_start + new_width
        canvas[y_start:y_end, x_start:x_end] = image_frame_resized

        # Combine frames horizontally
        combined_frame = np.hstack([text_frame, canvas])
        combined_frames.append(combined_frame)

    return combined_frames


def create_attention_visualization(
    attention_weights,
    sequences,
    processor,
    layer_idx=-1,
    fps=2,
    output_path="attention_visualization.mp4",
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
            attention_weights_step, current_tokens, processor
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
    checkpoint = (
        "/millcreek/home/sunil/r1_vlm/vlm-r1-message-decoding-words/checkpoint-2300"
    )

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

    dataset = load_dataset("sunildkumar/message-decoding-words-r1")["train"]

    examples = [ex for ex in dataset if ex["decoded_message"] == "SMART"]
    example = examples[0]

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
    generated_output = model.generate(
        **inputs,
        max_new_tokens=512,
        output_attentions=True,
        return_dict_in_generate=True,
    )
    print("Generation complete")

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
    output_path = f"combined_attention_visualization_layer{layer_idx}.mp4"
    imageio.imwrite(output_path, combined_frames, fps=2, codec="libx264")
