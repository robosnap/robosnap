import os
import cv2
import argparse
import re
import gradio as gr
import numpy as np
from pathlib import Path
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_predictor

MAIN_COLOR = (83, 160, 33)  # RGB


# =============================
# Utils
# =============================
def draw_points(img, points, labels):
    vis = img.copy()
    for (x, y), lbl in zip(points, labels):
        color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        cv2.circle(vis, (x, y), 12, (0, 0, 0), -1)
        cv2.circle(vis, (x, y), 9, color, -1)
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
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mask_area(mask: np.ndarray) -> float:
    return float(mask.sum())


def _mask_perimeter(mask: np.ndarray) -> float:
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


def read_video(path: Path):
    """
    Read video or single image.
    Returns: (frames: list of RGB arrays, fps: float, is_single_image: bool)
    """
    # Check if it's an image file first
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    if path.suffix.lower() in image_extensions:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
        return frames, 1.0, True  # fps = 1.0 for single image, is_single_image = True

    # Original video reading logic
    cap = cv2.VideoCapture(str(path))
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
    Image.fromarray(rgba, mode="RGBA").save(out_path)


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


# =============================
# Main
# =============================
def main(args):
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    input_path = Path(args.video)
    frames, video_fps, is_single_image = read_video(input_path)
    frame0 = frames[0]
    H, W = frame0.shape[:2]

    # >>> NEW: Detect if input is single image
    is_single_image = (len(frames) == 1)

    if is_single_image:
        print(f"[INFO] Single image mode detected: {input_path.name}")
    else:
        print(f"[INFO] Video mode detected: {len(frames)} frames, fps={video_fps}")

    # -----------------------------
    # SAM3 Video Predictor
    # -----------------------------
    predictor = build_sam3_video_predictor(checkpoint_path=args.ckpt)
    predictor.model.compile_model = False
    predictor.model = predictor.model.float().cuda()

    session_info = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(args.video),
    })
    session_id = session_info["session_id"]

    # -----------------------------
    # Python-side state
    # -----------------------------
    current_prompt = None
    current_obj_id = 0
    points_rel = []        # [(x_rel, y_rel)]
    point_labels = []      # [1 | 0]
    saved_objects = []

    # >>> CHANGED: keep last preview video path to show in UI
    last_preview_video = None

    # -----------------------------
    # UI
    # -----------------------------
    with gr.Blocks(title="SAM3 Interactive Video Segmentation") as demo:
        # Dynamic title based on mode
        if is_single_image:
            gr.Markdown("## SAM3 Interactive Image Segmentation (Single Image Mode)")
        else:
            gr.Markdown("## SAM3 Interactive Video Segmentation (Video Mode)")

        with gr.Row():
            img = gr.Image(value=frame0, interactive=True, label="Click to add points")
            # >>> NEW: Use Image component for preview in single image mode, Video for video mode
            if is_single_image:
                preview_img = gr.Image(label="Mask Preview", interactive=False)
            else:
                preview_img = gr.Video(label="Single Object Video Preview", autoplay=True, loop=True)

        prompt = gr.Textbox(label="Text Prompt", placeholder="e.g. personal computer")
        mode = gr.Radio(["positive", "negative"], value="positive", label="Point Type")

        with gr.Row():
            btn_confirm = gr.Button("✅ Confirm Prompt")
            btn_reset = gr.Button("🔄 Reset Points")
            btn_confirm_mask = gr.Button("🎬 Confirm Mask & Preview")
            btn_next_obj = gr.Button("➕ Save Masks & Next")
            btn_end = gr.Button("🛑 End", variant="primary")

        status = gr.Markdown("🟢 Ready")

        # -----------------------------
        # Callbacks
        # -----------------------------
        def confirm_prompt(prompt_text):
            nonlocal current_prompt, points_rel, point_labels

            prompt_text = (prompt_text or "").strip()
            if prompt_text == "":
                return gr.update(), "❌ Empty prompt"

            current_prompt = prompt_text
            points_rel.clear()
            point_labels.clear()

            # Check if this is xian mode
            is_interactive_area = (prompt_text.lower().replace(" ", "") == "interactivearea")

            resp = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": current_prompt,
            })

            mask = _safe_first_mask(resp.get("outputs", {}))
            vis = overlay_mask(frame0, mask)

            # Different hint for interactive area mode
            if is_interactive_area:
                return vis, f"✅ Prompt set: `{current_prompt}`. Please click to select the interactive area."
            else:
                return vis, f"✅ Prompt set: `{current_prompt}`. Now click to refine."

        def on_click(evt: gr.SelectData, point_mode: str):
            nonlocal points_rel, point_labels, current_prompt, current_obj_id

            if current_prompt is None:
                return gr.update(), "❌ Please confirm a prompt first."

            x_px, y_px = evt.index
            x_rel, y_rel = px_point_to_rel(x_px, y_px, W, H)

            points_rel.append((x_rel, y_rel))
            point_labels.append(1 if point_mode == "positive" else 0)

            resp = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "obj_id": current_obj_id,
                "points": points_rel,
                "point_labels": point_labels,
                "rel_coordinates": True,
            })

            mask = _safe_mask_for_obj(resp.get("outputs", {}), current_obj_id)
            vis = overlay_mask(frame0, mask)

            points_px = [rel_point_to_px(x, y, W, H) for (x, y) in points_rel]
            vis = draw_points(vis, points_px, point_labels)
            return vis, f"🟡 {len(points_rel)} points"

        def reset_points():
            nonlocal points_rel, point_labels
            points_rel.clear()
            point_labels.clear()
            vis = frame0.copy()
            return vis, "🔄 Points cleared"

        def confirm_mask_and_preview():
            """
            >>> CHANGED:
            - support single image mode: show static image preview instead of video
            - use forward+reverse propagation for video to avoid cache assert
            - build a preview video with overlay
            """
            nonlocal last_preview_video

            if current_prompt is None:
                return None, "❌ Please confirm a prompt first."
            if len(points_rel) == 0:
                return None, "❌ No points. Add at least 1 point to refine mask."

            # finalize prompt for this obj_id at frame0
            predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": current_prompt,
            })
            
            predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "obj_id": current_obj_id,
                "points": points_rel,
                "point_labels": point_labels,
                "rel_coordinates": True,
            })

            # >>> NEW: Handle single image mode vs video mode
            if is_single_image:
                # Single image mode: just show the mask overlay on frame0
                resp = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "obj_id": current_obj_id,
                    "points": points_rel,
                    "point_labels": point_labels,
                    "rel_coordinates": True,
                })
                mask = _safe_mask_for_obj(resp.get("outputs", {}), current_obj_id)
                vis = overlay_mask(frame0, mask)
                # Draw points on the preview
                points_px = [rel_point_to_px(x, y, W, H) for (x, y) in points_rel]
                vis = draw_points(vis, points_px, point_labels)
                return vis, "🎬 Preview ready (single image mode)."
            else:
                # Video mode: generate video preview with propagation
                # Write preview under /tmp so Gradio can return it safely.
                tmp_video = Path("/tmp") / f"sam3_preview_obj_{current_obj_id}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(str(tmp_video), fourcc, video_fps, (W, H))

                # >>> CHANGED: two-pass propagation
                for out in stream_propagate_two_passes(predictor, session_id, start_frame_index=0):
                    fidx = out.get("frame_index", None)
                    if fidx is None or fidx < 0 or fidx >= len(frames):
                        continue

                    mask = _safe_mask_for_obj(out.get("outputs", {}), current_obj_id)
                    if mask is None:
                        # still write raw frame to keep timing consistent
                        vw.write(cv2.cvtColor(frames[fidx], cv2.COLOR_RGB2BGR))
                        continue

                    vis = overlay_mask(frames[fidx], mask)
                    vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

                vw.release()
                last_preview_video = str(tmp_video)
                return last_preview_video, "🎬 Preview ready (forward+reverse done)."

        def save_and_next_object():
            """
            >>> CHANGED:
            - support single image mode: skip propagation, directly save frame0 mask
            - run forward+reverse propagation for video
            - save RGBA PNG per frame (object RGB, bg transparent)
            - if prompt is "interactive area", save background instead (mask inverted) to background folder
            - reset UI state for next object
            """
            nonlocal current_obj_id, current_prompt, points_rel, point_labels, saved_objects

            if current_prompt is None:
                return frame0, None, "❌ No active prompt."
            if len(points_rel) == 0:
                return frame0, None, "❌ No points. Confirm/Refine mask first."

            # Check if this is interactive area mode (more robust check)
            is_interactive_area = (current_prompt.lower().replace(" ", "") == "interactivearea")

            prompt_dir = out_dir / sanitize_prompt_name(current_prompt)
            obj_dir = prompt_dir
            ensure_dir(obj_dir)

            # ensure prompt is finalized
            predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": current_prompt,
            })
            predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "obj_id": current_obj_id,
                "points": points_rel,
                "point_labels": point_labels,
                "rel_coordinates": True,
            })

            # Get high-quality frame0 mask (this is the mask user clicked on - highest quality!)
            resp_frame0 = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "obj_id": current_obj_id,
                "points": points_rel,
                "point_labels": point_labels,
                "rel_coordinates": True,
            })
            frame0_mask = _safe_mask_for_obj(resp_frame0.get("outputs", {}), current_obj_id)

            # >>> NEW: Handle single image mode vs video mode differently
            if is_single_image:
                # Single image mode: just save the frame0 mask
                if is_interactive_area:
                    # Save background (mask inverted)
                    background_dir = out_dir / "background"
                    ensure_dir(background_dir)
                    if frame0_mask is not None:
                        save_rgba_background(frame0_mask, frames[0], background_dir / "000000.png")
                        print(f"[interactive_area] Saved background mask to background/000000.png")
                    saved_objects.append("background")
                    save_message = f"✅ Saved 1 background frame to `{background_dir}`."
                else:
                    # Save object mask
                    if frame0_mask is not None:
                        save_rgba_mask(frame0_mask, frames[0], obj_dir / "000000.png")
                        print(f"[single_image] Saved mask to {obj_dir / '000000.png'}")
                    saved_objects.append(obj_dir.name)
                    save_message = f"✅ Saved 1 mask to `{obj_dir}`."

                # Reset state for next object
                current_obj_id += 1
                current_prompt = None
                points_rel.clear()
                point_labels.clear()

                return frame0, None, f"{save_message} Ready for next object."
            
            # >>> Original video mode logic below <<<
            obj_video_path = obj_dir / "background_only.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(obj_video_path), fourcc, video_fps, (W, H))
            if not vw.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                vw = cv2.VideoWriter(str(obj_video_path), fourcc, video_fps, (W, H))

            masks_by_idx = {}
            for out in stream_propagate_forward(predictor, session_id, start_frame_index=0):
                fidx = out.get("frame_index", None)
                if fidx is None or fidx < 0 or fidx >= len(frames):
                    continue
                mask = _safe_mask_for_obj(out.get("outputs", {}), current_obj_id)
                if mask is None:
                    mask = np.zeros(frames[fidx].shape[:2], dtype=bool)
                masks_by_idx[fidx] = mask

            # write full background-only video (all frames)
            # >>> CHANGED: Use high-quality frame0 mask for first frame
            for fidx in range(len(frames)):
                if fidx == 0 and frame0_mask is not None:
                    mask = frame0_mask  # Use high-quality clicked mask for frame0
                else:
                    mask = masks_by_idx.get(
                        fidx, np.zeros(frames[fidx].shape[:2], dtype=bool)
                    )
                bg_only = frames[fidx].copy()
                bg_only[mask] = 0
                vw.write(cv2.cvtColor(bg_only, cv2.COLOR_RGB2BGR))

            sampled_indices = select_frames_by_quality(
                frames=frames,
                masks_by_idx=masks_by_idx,
                max_frames=args.max_frames,
            )

            saved_count = 0

            # >>> NEW: Handle interactive area case
            if is_interactive_area:
                # Create background folder
                background_dir = out_dir / "background"
                ensure_dir(background_dir)

                # >>> CHANGED: Save high-quality frame0 mask first (user clicked)
                if frame0_mask is not None:
                    save_rgba_background(frame0_mask, frames[0], background_dir / "000000.png")
                    print(f"[interactive_area] Saved high-quality frame0 mask to background/000000.png")
                else:
                    print(f"[interactive_area] Warning: frame0_mask is None, using propagated mask")

                # Save frames from index 1 onwards (skip frame0 since we already saved high-quality version)
                for fidx in range(1, len(frames)):
                    mask = masks_by_idx.get(
                        fidx, np.zeros(frames[fidx].shape[:2], dtype=bool)
                    )
                    # Save background (mask inverted): background = visible, interactive area = transparent
                    save_rgba_background(mask, frames[fidx], background_dir / f"{fidx:06d}.png")

                saved_count = len(frames)  # Count includes frame0

                # Generate background video using ffmpeg
                background_video_path = background_dir / "background_video.mp4"
                generate_background_video(background_dir, background_video_path, video_fps)

                saved_objects.append("background")
                save_message = f"✅ Saved {saved_count} background frames to `{background_dir}` and video `{background_video_path}`."
            else:
                # Original logic: save object masks
                for fidx in sampled_indices:
                    mask = masks_by_idx.get(
                        fidx, np.zeros(frames[fidx].shape[:2], dtype=bool)
                    )
                    # Save with original frame index (e.g., 000012.png) for proper alignment with depth estimation
                    save_rgba_mask(mask, frames[fidx], obj_dir / f"{fidx:06d}.png")
                    saved_count += 1

                saved_objects.append(obj_dir.name)
                save_message = f"✅ Saved {saved_count} masks to `{obj_dir}` and video `{obj_video_path}`."

            vw.release()

            # reset state for next object
            current_obj_id += 1
            current_prompt = None
            points_rel.clear()
            point_labels.clear()

            return (
                    frame0,
                    None,
                f"{save_message} Ready for next object.",
                )

        def on_end():
            print("Finished objects:", saved_objects)
            os._exit(0)

        # -----------------------------
        # Bind
        # -----------------------------
        btn_confirm.click(confirm_prompt, inputs=[prompt], outputs=[img, status])
        img.select(on_click, inputs=[mode], outputs=[img, status])
        btn_reset.click(reset_points, outputs=[img, status])

        # >>> NEW: Bind to correct preview component based on mode
        if is_single_image:
            btn_confirm_mask.click(confirm_mask_and_preview, inputs=None, outputs=[preview_img, status])
            btn_next_obj.click(save_and_next_object, inputs=None, outputs=[img, preview_img, status])
        else:
            btn_confirm_mask.click(confirm_mask_and_preview, inputs=None, outputs=[preview_img, status])
            btn_next_obj.click(save_and_next_object, inputs=None, outputs=[img, preview_img, status])

        btn_end.click(on_end, outputs=[status])

        demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            debug=args.debug,
            show_error=True,
            allowed_paths=[str(out_dir)],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--port", type=int, default=5100)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_frames", type=int, default=10)
    args = parser.parse_args()
    main(args)