import argparse
import sys
import traceback
from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve

from PIL import Image
from imgcat import imgcat

from reasoncv.opbase import DebuggingTextContext
from reasoncv.parser import RcvOpParser
from reasoncv.ops import available_ops


class ImageLoader:
    """Handles loading and caching of images from URLs or local paths"""

    def __init__(self, cache_dir: Path | str | None = None):
        """Initialize the image loader with a cache directory
        
        Args:
            cache_dir: Directory to cache downloaded images. If None, uses ~/.reasoncv/cache
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".reasoncv" / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download(self, url: str) -> Path:
        """Downloads an image from a URL and caches it locally
        
        Args:
            url: URL of the image to download
            
        Returns:
            Path to the cached image file
        """
        # Create a filename from the last part of the URL
        filename = url.split("/")[-1]
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filename += ".jpg"
        
        cache_path = self.cache_dir / filename
        if not cache_path.exists():
            print(f"Downloading {url} to {cache_path}")
            urlretrieve(url, cache_path)
        
        return cache_path

    def load(self, image_source: str) -> Tuple[Image.Image, Path]:
        """Loads an image from a URL or local path
        
        Args:
            image_source: URL or path to image file
            
        Returns:
            Tuple of (loaded PIL Image, path to image file)
            
        Raises:
            SystemExit: If image cannot be found/loaded
        """
        if image_source.startswith(("http://", "https://")):
            image_path = self._download(image_source)
        else:
            image_path = Path(image_source)
        
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            sys.exit(1)

        image = Image.open(image_path)
        return image, image_path


def read_multiline_input() -> str:
    """Reads multi-line input until </op> is seen
    
    Returns:
        The complete input including the </op> line
    """
    lines = []
    while True:
        try:
            line = input().rstrip()
            lines.append(line)
            if line.strip() == "</op>":
                break
        except EOFError:
            break
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Test ReasonCV ops interactively")
    parser.add_argument(
        "--image",
        type=str,
        default="https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        help="URL or path to image file",
    )
    args = parser.parse_args()

    # Set up image loading
    loader = ImageLoader()
    image, image_path = loader.load(args.image)
    
    context = DebuggingTextContext()
    op_parser = RcvOpParser()
    
    print(f"Loaded image: {image_path} (size={image.size})")
    imgcat(image)
    print("\nEnter ops in YAML format within <op> tags, e.g.:")
    print("<op>")
    print("function: object-detection")
    print("object_name: dog")
    print("threshold: 0.5")
    print("</op>")
    print(f"Available ops: {', '.join(available_ops())}")
    print("\nEnter 'quit' to exit")
    
    while True:
        try:
            print("\n> ", end="")
            user_input = read_multiline_input()
            
            if user_input.lower() in ("quit", "exit", "q"):
                break
            
            if not user_input:
                continue
            
            # Parse the op request using the existing parser
            try:
                op_requests = op_parser.parse(user_input)
                for op in op_requests:
                    op.run(image, context)
            except ValueError as e:
                print(f"Error: {e}")
                traceback.print_exc()
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main() 