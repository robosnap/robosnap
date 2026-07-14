"""Video, mask, and propagation helpers for the RoboSnap GUI."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image


MAIN_COLOR = (83, 160, 33)  # RGB
POINT_OUTER_RADIUS = 5
POINT_INNER_RADIUS = 3


def draw_points(img, points, labels):
    import cv2

    vis = img.copy()
    for (x, y), lbl in zip(points, labels):
        color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        cv2.circle(vis, (x, y), POINT_OUTER_RADIUS, (0, 0, 0), -1)
        cv2.circle(vis, (x, y), POINT_INNER_RADIUS, color, -1)
    return vis

def rel_point_to_px(x_rel, y_rel, W, H):
    return int(x_rel * W), int(y_rel * H)

def px_point_to_rel(x, y, W, H):
    return (
        float(np.clip(x / W, 0.0, 1.0)),
        float(np.clip(y / H, 0.0, 1.0)),
    )

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_prompt_name(text: str) -> str:
    name = (text or "").strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z._-]", "_", name)
    name = name.strip("._-")
    return name or "prompt"

def get_sampled_indices(num_frames: int, max_frames: int | None) -> list[int]:
    if not max_frames or max_frames <= 0 or num_frames <= max_frames:
        return list(range(num_frames))
    indices = [int(round(x)) for x in np.linspace(0, num_frames - 1, max_frames)]
    uniq = []
    for idx in indices:
        idx = min(max(idx, 0), num_frames - 1)
        if idx not in uniq:
            uniq.append(idx)
    if len(uniq) < max_frames:
        for idx in range(num_frames):
            if idx not in uniq:
                uniq.append(idx)
            if len(uniq) == max_frames:
                break
    return uniq

def _laplacian_var(gray: np.ndarray) -> float:
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _mask_area(mask: np.ndarray) -> float:
    return float(mask.sum())

def _mask_perimeter(mask: np.ndarray) -> float:
    import cv2

    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return float(sum(cv2.arcLength(c, True) for c in contours))

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0

def _frame_hist_feature(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    hist = []
    for ch in range(3):
        h, _ = np.histogram(rgb[:, :, ch], bins=bins, range=(0, 255), density=True)
        hist.append(h)
    feat = np.concatenate(hist, axis=0)
    return feat / (np.linalg.norm(feat) + 1e-8)

def select_frames_by_quality(
    frames: list[np.ndarray],
    masks_by_idx: dict[int, np.ndarray],
    max_frames: int,
) -> list[int]:
    import cv2

    num_frames = len(frames)
    if max_frames <= 0 or num_frames <= max_frames:
        return list(range(num_frames))

    # precompute median mask area for completeness check
    areas = [
        _mask_area(m)
        for m in masks_by_idx.values()
        if m is not None and m.size > 0 and m.any()
    ]
    median_area = float(np.median(areas)) if areas else 0.0

    # score each frame
    scores = []
    score_by_idx = {}
    prev_gray = None
    prev_mask = None
    for idx, rgb in enumerate(frames):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sharp = _laplacian_var(gray)
        motion = 0.0
        if prev_gray is not None:
            motion = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray)))
        prev_gray = gray

        mask = masks_by_idx.get(idx, None)
        if mask is None:
            mask_area = 0.0
            mask_perim = 0.0
            mask_iou = 0.0
            completeness = 0.0
        else:
            mask_area = _mask_area(mask)
            mask_perim = _mask_perimeter(mask)
            mask_iou = _mask_iou(mask, prev_mask) if prev_mask is not None else 0.0
            if median_area > 0:
                completeness = min(mask_area / (median_area + 1e-6), 1.0)
            else:
                completeness = 0.0
        prev_mask = mask

        # normalize simple terms
        score = (
            0.35 * (sharp / (sharp + 1e-6))
            + 0.25 * (1.0 / (1.0 + motion))
            + 0.15 * (mask_area / (mask_area + 1e-6))
            + 0.1 * mask_iou
            + 0.2 * completeness
        )
        scores.append((idx, score))
        score_by_idx[idx] = {
            "score": score,
            "sharp": sharp,
            "motion": motion,
            "mask_area": mask_area,
            "mask_perim": mask_perim,
            "mask_iou": mask_iou,
            "completeness": completeness,
        }

    # take top candidates, then diversify
    scores.sort(key=lambda x: x[1], reverse=True)
    top_k = min(len(scores), max_frames * 5)
    candidates = [idx for idx, _ in scores[:top_k]]
    print(
        f"[sampling] total_frames={len(scores)} top_k={top_k} max_frames={max_frames}"
    )
    for rank, (idx, _) in enumerate(scores[:top_k], start=1):
        metric = score_by_idx[idx]
        print(
            f"[sampling] rank={rank:02d} frame={idx:04d} "
            f"score={metric['score']:.4f} "
            f"sharp={metric['sharp']:.4f} motion={metric['motion']:.4f} "
            f"mask_area={metric['mask_area']:.4f} mask_perim={metric['mask_perim']:.4f} "
                f"mask_iou={metric['mask_iou']:.4f} completeness={metric['completeness']:.4f}"
        )
    feats = {idx: _frame_hist_feature(frames[idx]) for idx in candidates}

    selected = []
    if candidates:
        selected.append(candidates[0])

    while len(selected) < max_frames and len(selected) < len(candidates):
        best_idx = None
        best_min_dist = -1.0
        for idx in candidates:
            if idx in selected:
                continue
            feat = feats[idx]
            min_dist = min(
                float(np.linalg.norm(feat - feats[sidx])) for sidx in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)

    selected = sorted(selected)
    print(f"[sampling] selected_frames={selected}")
    return selected

def read_video(path: Path, lossless: bool = True):
    """
    Read video or single image.
    Returns: (frames: list of RGB arrays, fps: float, is_single_image: bool)
    
    Args:
        lossless: If True, use ffmpeg for lossless frame extraction
    """
    import cv2

    # Check if it's an image file first
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    if path.suffix.lower() in image_extensions:
        # For PNG/TIFF, use IMREAD_UNCHANGED to preserve quality
        if path.suffix.lower() in {'.png', '.tiff', '.tif'}:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
            # Handle RGBA
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # For lossy formats, read with full quality
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames = [img]
        return frames, 1.0, True  # fps = 1.0 for single image, is_single_image = True

    # Video reading - use ffmpeg for lossless extraction if requested
    if lossless:
        try:
            import subprocess
            # Use ffmpeg to extract frames as PNG (lossless)
            # This gives us full original quality
            cmd = [
                'ffmpeg', '-i', str(path),
                '-vf', 'scale=iw:ih',  # Keep original resolution
                '-pix_fmt', 'rgb24',   # RGB format
                '-vcodec', 'png',      # Lossless codec
                '-fps_mode', 'passthrough',  # Don't drop frames
                '-f', 'image2pipe',
                '-'
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            import io
            from PIL import Image
            frames = []
            
            # Read frames from pipe
            chunk = process.stdout.read(8 * 1024 * 1024)  # 8MB chunks
            while chunk:
                try:
                    # Try to find and decode PNG images
                    img = Image.open(io.BytesIO(chunk))
                    if img.format == 'PNG':
                        frames.append(np.array(img.convert('RGB')))
                    # Continue reading
                    chunk = process.stdout.read(8 * 1024 * 1024)
                except Exception:
                    chunk = process.stdout.read(8 * 1024 * 1024)
                    continue
                    
            process.terminate()
            
            if frames:
                # Get FPS from video metadata
                cmd_fps = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                          '-show_entries', 'stream=avg_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)]
                result = subprocess.run(cmd_fps, capture_output=True, text=True)
                fps_str = result.stdout.strip()
                if fps_str and '/' in fps_str:
                    num, denom = fps_str.split('/')
                    fps = float(num) / float(denom)
                else:
                    fps = float(fps_str) if fps_str else 30.0
                return frames, fps, False
        except Exception as e:
            print(f"[WARN] FFmpeg extraction failed: {e}, falling back to OpenCV")

    # Fallback: Original video reading logic with improved settings
    cap = cv2.VideoCapture(str(path))
    
    # Set CAP_PROP for maximum quality reading
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 20.0
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to read video: {path}")
    return frames, float(fps), False  # is_single_image = False

def overlay_mask(img, mask, alpha=0.5):
    vis = img.copy()
    if mask is not None:
        # mask expected bool HxW
        vis[mask] = (vis[mask] * (1 - alpha) + np.array(MAIN_COLOR) * alpha).astype(np.uint8)
    return vis

def create_video_writer(path, fps, width, height):
    """
    Create VideoWriter with multiple codec fallbacks.
    Returns: VideoWriter or None if all fail
    """
    import cv2

    # Try different codecs in order of preference
    codecs_to_try = [
        ('mp4v', '.mp4'),    # MPEG-4 Part 2 - most compatible
        ('XVID', '.avi'),    # Xvid codec
        ('X264', '.mp4'),    # H.264 
        ('avc1', '.mp4'),    # H.264 (Apple)
        ('H264', '.mp4'),    # H.264
        ('MJPG', '.avi'),    # Motion JPEG
    ]
    
    # Try output path with different extensions
    base_path = str(path).rsplit('.', 1)[0]
    
    for codec_ext, (fourcc_str, ext) in enumerate(codecs_to_try):
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            test_path = f"{base_path}_temp{ext}"
            vw = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
            if vw.isOpened():
                # Move to final path
                vw.release()
                import os
                if os.path.exists(test_path):
                    os.remove(test_path)
                final_path = f"{base_path}{ext}"
                vw = cv2.VideoWriter(final_path, fourcc, fps, (width, height))
                if vw.isOpened():
                    print(f"[INFO] Using codec: {fourcc_str}")
                    return vw, final_path
                vw.release()
        except Exception as e:
            print(f"[WARN] Codec {fourcc_str} failed: {e}")
            continue
    
    # Last resort: try OpenCV's default
    try:
        vw = cv2.VideoWriter(str(path), -1, fps, (width, height))
        if vw.isOpened():
            return vw, str(path)
    except:
        pass
    
    print("[ERROR] All video codecs failed!")
    return None, str(path)

def save_rgba_mask(mask, rgb_frame: np.ndarray, out_path: Path):
    """
    Save RGBA where:
    - object pixels: original RGB
    - background: transparent
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # >>> CHANGED: force bool mask safely
    if mask is None:
        return
    if isinstance(mask, np.ndarray):
        mask_bool = mask.astype(bool)
    else:
        # in case it's torch tensor (rare in this path)
        mask_bool = mask.detach().cpu().numpy().astype(bool)

    alpha = (mask_bool.astype(np.uint8) * 255)
    rgba = np.dstack([rgb_frame, alpha])
    
    # Save as PNG (lossless) with maximum compression for quality
    img = Image.fromarray(rgba, mode="RGBA")
    if out_path.suffix.lower() == '.png':
        img.save(out_path, compress_level=0)  # No compression = maximum quality
    else:
        img.save(out_path, quality=100, subsampling=0)  # JPEG max quality

def save_rgba_background(mask, rgb_frame: np.ndarray, out_path: Path):
    """
    Save RGBA where:
    - background pixels (mask=False): original RGB
    - interactive area (mask=True): transparent
    This is the inverse of save_rgba_mask
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mask is None:
        return
    if isinstance(mask, np.ndarray):
        mask_bool = mask.astype(bool)
    else:
        mask_bool = mask.detach().cpu().numpy().astype(bool)

    # Invert mask: background = True, interactive area = False
    background_mask = ~mask_bool
    alpha = (background_mask.astype(np.uint8) * 255)
    rgba = np.dstack([rgb_frame, alpha])
    Image.fromarray(rgba, mode="RGBA").save(out_path)

def generate_background_video(frames_dir: Path, output_video: Path, fps: float, pattern: str = "*.png"):
    """
    Use ffmpeg to generate video from PNG frames.
    First processes RGBA PNGs: transparent pixels (alpha=0) become black.
    Then uses ffmpeg to encode.
    """
    import subprocess
    from PIL import Image
    import numpy as np

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: ffmpeg not found, skipping video generation")
        return False

    # Get list of PNG files sorted by frame index
    png_files = sorted(frames_dir.glob(pattern))
    if not png_files:
        print(f"Warning: no PNG files found in {frames_dir}")
        return False

    # Check if files are RGBA and need conversion
    needs_conversion = False
    sample_img = Image.open(png_files[0])
    if sample_img.mode == 'RGBA':
        needs_conversion = True

    if needs_conversion:
        print("Converting RGBA to RGB with black mask...")
        temp_dir = frames_dir / "_temp_rgb"
        temp_dir.mkdir(exist_ok=True)

        for i, f in enumerate(png_files):
            img = Image.open(f).convert('RGBA')
            arr = np.array(img)
            # Make transparent pixels (alpha=0) black
            arr[arr[:,:,3] == 0, :3] = 0
            # Save as RGB
            rgb_img = Image.fromarray(arr[:,:,:3], 'RGB')
            rgb_img.save(temp_dir / f"frame_{i:06d}.rgb.png")

        input_pattern = str(temp_dir / "frame_%06d.rgb.png")
    else:
        input_pattern = str(frames_dir / pattern)

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        str(output_video)
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"Generated background video: {output_video}")
        # Clean up temp files
        if needs_conversion:
            import shutil
            shutil.rmtree(temp_dir)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating video: {e}")
        print(f"ffmpeg stderr: {e.stderr}")
        return False

def _safe_first_mask(outputs_dict):
    """
    >>> CHANGED: unify mask extraction; avoid `if masks:` numpy truth-value
    Returns: mask (H,W bool) or None
    """
    masks = outputs_dict.get("out_binary_masks", None)
    if masks is None or len(masks) == 0:
        return None
    m = masks[0]
    if isinstance(m, np.ndarray):
        return m.astype(bool)
    return m.detach().cpu().numpy().astype(bool)

def _safe_mask_for_obj(outputs_dict, obj_id: int):
    """
    Returns mask (H,W bool) for a specific obj_id, or None if not found.
    """
    masks = outputs_dict.get("out_binary_masks", None)
    obj_ids = outputs_dict.get("out_obj_ids", None)
    if masks is None or obj_ids is None or len(masks) == 0:
        return None
    obj_ids = np.array(obj_ids)
    match = np.where(obj_ids == obj_id)[0]
    if len(match) == 0:
        return None
    m = masks[int(match[0])]
    if isinstance(m, np.ndarray):
        return m.astype(bool)
    return m.detach().cpu().numpy().astype(bool)

def stream_propagate_two_passes(predictor, session_id, start_frame_index=0):
    """
    >>> CHANGED: fix cached_frame_outputs error
    Do forward pass first (populate cache), then reverse pass.
    Yields dicts like predictor.handle_stream_request output.
    """
    # Pass 1: forward (normal propagation populates cache)
    for out in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
        "start_frame_index": start_frame_index,
        "propagation_direction": "forward",
    }):
        yield out

    # Pass 2: reverse (uses cache)
    for out in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
        "start_frame_index": start_frame_index,
        "propagation_direction": "backward",
    }):
        yield out

def stream_propagate_forward(predictor, session_id, start_frame_index=0):
    for out in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
        "start_frame_index": start_frame_index,
        "propagation_direction": "forward",
    }):
        yield out
