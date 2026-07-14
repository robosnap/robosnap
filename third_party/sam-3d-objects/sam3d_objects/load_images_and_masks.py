"""
Load multi-view data from specified path using RGBA-only masks.
Each view is a single RGBA PNG. RGB is the image, alpha is the mask.
The input folder is expected to be a single object folder with 0.png, 1.png, ...
"""
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from loguru import logger


def load_image_and_mask_from_rgba(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.open(path)
    img_array = np.array(img).astype(np.uint8)
    if img.mode == "RGBA" and img_array.ndim == 3 and img_array.shape[2] >= 4:
        image = img_array[..., :3]
        mask = img_array[..., 3] > 0
    elif img.mode == "RGB":
        image = img_array
        mask = np.ones((img_array.shape[0], img_array.shape[1]), dtype=bool)
    else:
        # fallback: treat as grayscale
        if img_array.ndim == 2:
            image = np.stack([img_array] * 3, axis=-1)
        else:
            image = img_array[..., :3]
        mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
    return image, mask


def load_images_and_masks(
    images_and_masks_dir: Path,
    image_names: list = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load multi-view data from a folder of RGBA PNGs.
    Each PNG encodes both image (RGB) and mask (alpha).
    
    Args:
        images_and_masks_dir: Path to RGBA views folder
        image_names: Optional list of image names (without extension) to load.
                    If not specified, all images in the directory will be loaded.
        
    Returns:
        images: List of images (numpy arrays)
        masks: List of masks (numpy arrays, bool format)
    """
    if not images_and_masks_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {images_and_masks_dir}")
    
    if not images_and_masks_dir.is_dir():
        raise ValueError(f"Path is not a directory: {images_and_masks_dir}")
    
    # Collect all RGBA PNG files
    image_files = list(images_and_masks_dir.glob("*.png"))
    
    # Filter by image_names if specified
    if image_names is not None and len(image_names) > 0:
        image_files = [f for f in image_files if f.stem in image_names]
        logger.info(f"Filtered to {len(image_files)} images based on image_names: {image_names}")

    # Sort by filename with natural number ordering
    # This ensures "9.png" comes after "8.jpg", not before "0.jpg"
    def natural_sort_key(path):
        """Sort key that handles numeric filenames correctly."""
        stem = path.stem
        # Try to extract leading number for sorting
        try:
            return (0, int(stem), stem)  # Numeric names first, sorted numerically
        except ValueError:
            return (1, 0, stem)  # Non-numeric names after, sorted alphabetically

    image_files = sorted(image_files, key=natural_sort_key)
    image_names = [f.stem for f in image_files]
    logger.info(f"Auto-detected {len(image_names)} images: {image_names}")
    
    images = []
    masks = []
    
    for image_name in image_names:
        image_path = images_and_masks_dir / f"{image_name}.png"
        try:
            if not image_path.exists():
                logger.warning(f"RGBA file not found for '{image_name}', skipping")
                continue
            image, mask = load_image_and_mask_from_rgba(image_path)
            
            images.append(image)
            masks.append(mask)
            
            logger.info(f"Loaded '{image_name}': image={image.shape}, mask={mask.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load '{image_name}': {e}")
            continue
    
    if len(images) == 0:
        raise ValueError(f"No valid images and masks found in {images_and_masks_dir}")
    
    logger.info(f"Successfully loaded {len(images)} images")
    return images, masks


def load_images_and_masks_from_path(
    input_path: Path,
    image_names: list = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load multi-view data from specified path using RGBA-only masks.
    If input_path has no PNGs but contains exactly one subfolder with PNGs,
    that subfolder will be used automatically.
    
    Args:
        input_path: Input path
        image_names: Optional list of image names (without extension) to load.
                   If not specified, all images in the path will be loaded.
        
    Returns:
        images: List of images
        masks: List of masks
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_path}")
    
    has_png = any(p.suffix.lower() == ".png" for p in input_path.iterdir() if p.is_file())
    if not has_png:
        subdirs = [
            d for d in input_path.iterdir()
            if d.is_dir() and any(p.suffix.lower() == ".png" for p in d.iterdir() if p.is_file())
        ]
        if len(subdirs) == 1:
            input_path = subdirs[0]
        elif len(subdirs) > 1:
            raise ValueError(
                "Input path contains multiple object folders. "
                "Please run inference on the base dir to process all objects."
            )
    logger.info(f"Loading RGBA views from: {input_path}")
    return load_images_and_masks(input_path, image_names=image_names)

