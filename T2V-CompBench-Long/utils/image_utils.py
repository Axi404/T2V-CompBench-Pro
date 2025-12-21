import requests
from PIL import Image
from io import BytesIO
from .transforms_utils import Compose, RandomResize, ToTensor, Normalize


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def load_and_process_image(image_path: str) -> tuple[Image.Image, "torch.Tensor"]:
    """
    Load and process an image for model input.
    
    Args:
        image_path: Path to the image file.
    
    Returns:
        A tuple of (PIL Image, processed tensor).
    """
    # Use with statement to ensure file handle is properly closed
    with Image.open(image_path) as img:
        image_pil = img.convert("RGB").copy()  # copy() to detach from file handle

    transform = Compose(
        [
            RandomResize([800], max_size=1333),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image
