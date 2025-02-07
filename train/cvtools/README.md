# CV Tools

An environment for "Reasoning VLMs" (RVLMs) which are able to interact with visual input data by asking for
things like:

- Zoom in to a particular area of the image.  Get more tokens to see parts of the image more clearly.
- Run external models such as:
    - Object detection
    - OCR
    - Depth / normal estimation
    - Segmentation
    - Keypoint detection / pose estimation

## Installation

```bash
uv sync
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Usage

### Interactive Text Trial

The `texttrial.py` script lets you test ReasonCV ops interactively. You can provide an image URL or local file path, and then enter ops in YAML format to process the image.

Run with the default test image:

```bash
uv run python ./reasoncv/texttrial.py
```

(You can specify your own image by passing the `--image` flag.)


Example ops you can try:

```yaml
# Detect objects
<op>
function: object-detection
object_name: dog
threshold: 0.5
</op>

# Zoom into a region
<op>
function: zoom
center: [0.5, 0.5]
size: [0.3, 0.3]
</op>

# Estimate depth
<op>
function: depth-estimation
</op>
```

Each op will process the image and display the results inline in your terminal.

## RCV-Ops

Each operation is called an "RCV-Op" and can be requested by the model inside its `<think>` section. They are initiated by the reasoning step outputting the `<op>` tag and then a YAML-formatted string that describes the operation.

Note that image coordinates are always specified in normalized coordinates (0 to 1), with the upper left corner being (0, 0) and the lower right corner being (1, 1) for the original input image.


## Testing

```bash
uv run pytest
```

