import os
from articulate_tool_manager import ArticulateToolManager, _clear_socks_proxy_env, _subprocess_env_for_python
from sam3d_bridge import generate_3d_meshes_for_all_objects as _generate_3d_meshes_for_all_objects
from viewer_html import _get_articulate_viewer_html
from media_utils import (
    _safe_first_mask,
    _safe_mask_for_obj,
    create_video_writer,
    draw_points,
    ensure_dir,
    generate_background_video,
    overlay_mask,
    px_point_to_rel,
    read_video,
    rel_point_to_px,
    sanitize_prompt_name,
    save_rgba_background,
    save_rgba_mask,
    select_frames_by_quality,
    stream_propagate_forward,
    stream_propagate_two_passes,
)




_clear_socks_proxy_env()
import argparse
import re
import shutil
from datetime import datetime, timezone
import re
import json
import base64
import zipfile
import gradio as gr
import numpy as np
from pathlib import Path
import torch
from PIL import Image
import subprocess

from sam3.model_builder import build_sam3_video_predictor
from articulate_manager import ArticulateObjectManager

ROBOSNAP_ROOT = Path(
    os.environ.get("ROBOSNAP_ROOT", Path(__file__).resolve().parents[3])
).expanduser().resolve()
CHECKPOINT_DIR = Path(
    os.environ.get("CHECKPOINT_DIR", ROBOSNAP_ROOT / "checkpoints")
).expanduser().resolve()

PY_ASSET = os.environ.get("PY_ASSET", "python")
PY_ARTICULATE = os.environ.get("PY_ARTICULATE", os.environ.get("PY_P3SAM", "python"))
SAM3_CKPT = os.environ.get("SAM3_CKPT", str(CHECKPOINT_DIR / "sam3" / "sam3.pt"))
SAM3D_DIR = Path(
    os.environ.get("SAM3D_DIR", ROBOSNAP_ROOT / "third_party" / "sam-3d-objects")
).expanduser().resolve()
SAM3D_CONFIG = os.environ.get(
    "SAM3D_CONFIG",
    str(CHECKPOINT_DIR / "sam-3d-objects" / "pipeline.yaml"),
)
ARTICULATE_APP = os.environ.get(
    "ARTICULATE_APP",
    os.environ.get("P3SAM_APP", str(ROBOSNAP_ROOT / "third_party" / "Hunyuan3D-Part" / "P3-SAM" / "demo" / "app.py")),
)
ARTICULATE_CKPT = os.environ.get(
    "ARTICULATE_CKPT",
    os.environ.get("P3SAM_CKPT", str(CHECKPOINT_DIR / "articulate" / "articulate.safetensors")),
)
ARTICULATE_BASE_PORT = int(os.environ.get("ARTICULATE_BASE_PORT", os.environ.get("P3SAM_BASE_PORT", "8180")))
EXTRA_ALLOWED_ROOTS = [
    str(Path(p).expanduser().resolve())
    for p in os.environ.get("GRADIO_ALLOWED_ROOTS", "").split(os.pathsep)
    if p
]
CURRENT_INPUT_VIDEO = None






def apply_runtime_config(args):
    """Apply CLI/env overrides so the release app is independent of local paths."""
    global PY_ASSET, PY_ARTICULATE, SAM3D_DIR, SAM3D_CONFIG, ARTICULATE_APP, ARTICULATE_CKPT, EXTRA_ALLOWED_ROOTS

    PY_ASSET = args.asset_python
    PY_ARTICULATE = args.articulate_python
    SAM3D_DIR = Path(args.asset_dir).expanduser().resolve()
    SAM3D_CONFIG = args.asset_config
    ARTICULATE_APP = args.articulate_app
    ARTICULATE_CKPT = args.articulate_ckpt
    if args.articulate_public_url_template:
        os.environ["ARTICULATE_PUBLIC_URL_TEMPLATE"] = args.articulate_public_url_template
        os.environ["P3SAM_PUBLIC_URL_TEMPLATE"] = args.articulate_public_url_template
    if args.allowed_root:
        EXTRA_ALLOWED_ROOTS.extend(
            str(Path(p).expanduser().resolve()) for p in args.allowed_root
        )
    _articulate_tool_manager.configure(
        python=PY_ARTICULATE,
        app=ARTICULATE_APP,
        ckpt=ARTICULATE_CKPT,
        base_port=args.articulate_base_port,
    )

# Global state for 3D generation progress
_GEN_STATE = {"running": False, "logs": "", "done": False}


def _unique_existing_paths(paths):
    result = []
    seen = set()
    for path in paths:
        if not path:
            continue
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            continue
        path_str = str(resolved)
        if path_str in seen:
            continue
        seen.add(path_str)
        result.append(path_str)
    return result


def _parse_basic_auth(auth_value):
    if not auth_value:
        return None

    auth_entries = []
    for entry in auth_value.split(','):
        entry = entry.strip()
        if not entry:
            continue
        if ':' not in entry:
            raise ValueError('GRADIO_AUTH entries must use username:password')
        username, password = entry.split(':', 1)
        if not username or not password:
            raise ValueError('GRADIO_AUTH username and password cannot be empty')
        auth_entries.append((username, password))

    if not auth_entries:
        return None
    if len(auth_entries) == 1:
        return auth_entries[0]
    return auth_entries


def _build_results_zip(source_dir: Path, download_dir: Path) -> Path:
    source_dir = source_dir.expanduser().resolve()
    download_dir = download_dir.expanduser().resolve()
    ensure_dir(download_dir)

    files = []
    for file_path in sorted(source_dir.rglob('*')):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(source_dir)
        if any(part.startswith('.') for part in rel_path.parts):
            continue
        files.append((file_path, rel_path))

    if not files:
        raise FileNotFoundError(f'No result files found under {source_dir}')

    zip_path = download_dir / f'{source_dir.name}_results.zip'
    tmp_path = download_dir / f'.{source_dir.name}_results.tmp.zip'
    with zipfile.ZipFile(tmp_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path, rel_path in files:
            zf.write(file_path, rel_path.as_posix())
    tmp_path.replace(zip_path)
    return zip_path


def generate_3d_meshes_for_all_objects(
    out_dir: Path,
    config_path: str = None,
    max_frames: int = 20,
    progress_callback=None,
):
    return _generate_3d_meshes_for_all_objects(
        out_dir,
        config_path,
        max_frames=max_frames,
        progress_callback=progress_callback,
        py_asset=PY_ASSET,
        sam3d_dir=SAM3D_DIR,
        default_config=SAM3D_CONFIG,
        source_video=CURRENT_INPUT_VIDEO,
        _subprocess_env_for_python=_subprocess_env_for_python,
    )


# =============================
# Articulate Tool Manager (for iframe integration)
# =============================



# Global articulate manager instance
_articulate_tool_manager = ArticulateToolManager(python=PY_ARTICULATE, app=ARTICULATE_APP, ckpt=ARTICULATE_CKPT, base_port=ARTICULATE_BASE_PORT)

# Store the Gradio server URL for iframe reference
_gradio_server_url = None

def set_graceful_exit():
    """Signal all articulate servers to stop gracefully"""
    _articulate_tool_manager.stop_all()


# =============================
# Utils - Mesh Generation
# =============================

# =============================
# 3D Generation Functions (Run after End button)
# =============================



# Note: generate_object_meshes is no longer needed - generate_3d_meshes_for_all_objects calls run_inference.py directly


# Note: run_scale_extraction is no longer needed - run_inference.py --compose_scene handles everything


# Note: generate_all_meshes is no longer needed - generate_3d_meshes_for_all_objects uses run_inference.py directly


# =============================
# Utils
# =============================














































# =============================
# Main
# =============================
def main(args):
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)
    download_dir = out_dir.parent / "downloads"

    input_path = Path(args.video).expanduser().resolve()
    global CURRENT_INPUT_VIDEO
    CURRENT_INPUT_VIDEO = input_path
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
    # Interactive segmentation predictor
    # -----------------------------
    predictor = build_sam3_video_predictor(checkpoint_path=args.ckpt)
    predictor.model.compile_model = False
    predictor.model = predictor.model.float().cuda()

    session_info = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(input_path),
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
    delete_cache = None
    if args.delete_cache_frequency > 0 and args.delete_cache_age > 0:
        delete_cache = (args.delete_cache_frequency, args.delete_cache_age)

    public_demo_root = out_dir.parent

    with gr.Blocks(title="Interactive Segmentation", delete_cache=delete_cache) as demo:
        # Dynamic title based on mode
        if is_single_image:
            gr.Markdown("## Interactive Image Segmentation (Single Image Mode)")
        else:
            gr.Markdown("## Interactive Video Segmentation (Video Mode)")

        if args.public_demo:
            with gr.Row():
                upload_video = gr.File(
                    label="Upload Video",
                    file_types=["video"],
                    type="filepath",
                )
                btn_load_upload = gr.Button("Load Uploaded Video")
        else:
            upload_video = None
            btn_load_upload = None

        with gr.Row():
            # Use display=True to show full resolution without client-side scaling
            img = gr.Image(
                value=frame0, 
                interactive=True, 
                label="Click to add points",
                image_mode="RGB",
                type="numpy"
            )
            # >>> NEW: Use Image component for preview in single image mode, Video for video mode
            if is_single_image:
                preview_img = gr.Image(
                    label="Mask Preview", 
                    interactive=False,
                    image_mode="RGB",
                    type="numpy"
                )
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

        with gr.Row():
            btn_download_results = gr.Button("⬇️ Prepare Results Download")
            results_download = gr.File(
                label="Results Zip",
                interactive=False,
                visible=False,
            )

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

            # Check if this is interactive area mode
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

        def load_uploaded_video(uploaded_path):
            nonlocal input_path, frames, video_fps, is_single_image, frame0, H, W, session_id
            nonlocal current_prompt, current_obj_id, saved_objects, last_preview_video
            nonlocal out_dir, download_dir
            global CURRENT_INPUT_VIDEO

            if not uploaded_path:
                return gr.update(), gr.update(), "❌ No uploaded video selected."

            uploaded_path = Path(uploaded_path).expanduser().resolve()
            if not uploaded_path.exists():
                return gr.update(), gr.update(), f"❌ Uploaded file does not exist: `{uploaded_path}`"

            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", uploaded_path.stem).strip("._") or "upload"
            session_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{safe_stem}"
            upload_dir = public_demo_root / "uploads"
            session_root = public_demo_root / "sessions" / session_name
            ensure_dir(upload_dir)
            ensure_dir(session_root)

            suffix = uploaded_path.suffix or ".mp4"
            stored_video = upload_dir / f"{session_name}{suffix}"
            shutil.copy2(uploaded_path, stored_video)

            try:
                new_frames, new_video_fps, new_is_single_image = read_video(stored_video)
            except Exception as exc:
                return gr.update(), gr.update(), f"❌ Failed to read uploaded video: {exc}"

            if len(new_frames) <= 1:
                return gr.update(), gr.update(), "❌ Please upload a video file with multiple frames."

            input_path = stored_video
            CURRENT_INPUT_VIDEO = input_path
            frames = new_frames
            video_fps = new_video_fps
            is_single_image = new_is_single_image
            frame0 = frames[0]
            H, W = frame0.shape[:2]

            out_dir = session_root / "multi_mask"
            download_dir = session_root / "downloads"
            ensure_dir(out_dir)
            ensure_dir(download_dir)

            session_info_new = predictor.handle_request({
                "type": "start_session",
                "resource_path": str(input_path),
            })
            session_id = session_info_new["session_id"]

            current_prompt = None
            current_obj_id = 0
            points_rel.clear()
            point_labels.clear()
            saved_objects.clear()
            last_preview_video = None

            return (
                frame0,
                None,
                f"✅ Loaded uploaded video `{stored_video.name}`. Outputs will be saved under `{out_dir}`.",
            )

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
                # Use robust video writer with multiple codec fallbacks
                result = create_video_writer(tmp_video, video_fps, W, H)
                vw, actual_video_path = result
                if vw is None:
                    return None, "❌ Failed to create video writer. Check ffmpeg installation."
                last_preview_video = str(actual_video_path)

                import cv2

                # >>> CHANGED: two-pass propagation with fallback to forward-only
                try:
                    propagation_gen = stream_propagate_two_passes(predictor, session_id, start_frame_index=0)
                    mode = "forward+reverse"
                except KeyError as e:
                    if "obj_id_to_score" in str(e):
                        print("[WARN] Two-pass failed, falling back to forward-only propagation")
                        propagation_gen = stream_propagate_forward(predictor, session_id, start_frame_index=0)
                        mode = "forward-only"
                    else:
                        raise

                for out in propagation_gen:
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
                return last_preview_video, f"🎬 Preview ready ({mode} done)."

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
                    # Create background folder (same as video mode)
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
            # Create folders for all masks and top* masks
            max_frames = args.max_frames
            all_mask_dir = obj_dir / "all_mask"
            top_mask_dir = obj_dir / f"top{max_frames}_mask"
            ensure_dir(all_mask_dir)
            ensure_dir(top_mask_dir)
            
            masks_by_idx = {}
            for out in stream_propagate_forward(predictor, session_id, start_frame_index=0):
                fidx = out.get("frame_index", None)
                if fidx is None or fidx < 0 or fidx >= len(frames):
                    continue
                mask = _safe_mask_for_obj(out.get("outputs", {}), current_obj_id)
                if mask is None:
                    mask = np.zeros(frames[fidx].shape[:2], dtype=bool)
                masks_by_idx[fidx] = mask

            # >>> NEW: Replace frame 0 with high-quality clicked mask for quality selection
            if frame0_mask is not None:
                masks_by_idx[0] = frame0_mask

            # write full background-only video (all frames) and save all masks
            for fidx in range(len(frames)):
                mask = masks_by_idx.get(fidx, np.zeros(frames[fidx].shape[:2], dtype=bool))
                
                # Save ALL masks to all_mask folder
                save_rgba_mask(mask, frames[fidx], all_mask_dir / f"{fidx:06d}.png")

            sampled_indices = select_frames_by_quality(
                frames=frames,
                masks_by_idx=masks_by_idx,
                max_frames=args.max_frames,
            )
            
            # >>> NEW: Save TOP masks to top*_mask folder
            for i, fidx in enumerate(sampled_indices[:max_frames]):
                # masks_by_idx now contains high-quality frame0_mask for frame 0
                mask = masks_by_idx.get(fidx, np.zeros(frames[fidx].shape[:2], dtype=bool))
                save_rgba_mask(mask, frames[fidx], top_mask_dir / f"{i:02d}_{fidx:06d}.png")

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
                # New logic: all masks saved in all_mask folder, top in top*_mask folder
                # saved_count = total frames saved to all_mask
                saved_count = len(frames)

                saved_objects.append(obj_dir.name)
                save_message = f"✅ Saved {saved_count} masks to `{all_mask_dir}`, top{max_frames} to `{top_mask_dir}`."

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
            """
            End video segmentation and transition to preview mode.
            """
            print("Finished objects:", saved_objects)
            # Instead of exiting, return message to transition to articulate mode
            return "🎯 Segmentation complete! Now you can preview the 3D assets generated below and get ready to choose the ones you want to segment as articulated objects."

        def prepare_results_download():
            try:
                zip_path = _build_results_zip(out_dir, download_dir)
            except FileNotFoundError:
                return gr.update(value=None, visible=False), "❌ No saved results yet. Save at least one mask first."
            except Exception as exc:
                return gr.update(value=None, visible=False), f"❌ Failed to prepare download: {exc}"
            return (
                gr.update(value=str(zip_path), visible=True),
                f"✅ Results ready for download: `{zip_path}`",
            )

        def on_enter_articulate_mode():
            """
            Transition to articulate mode - this will be handled by the UI state.
            """
            # Update state to indicate we're in articulate mode
            return "✅ Now in Articulate Object Annotation Mode. Select an object to annotate joints."

        # -----------------------------
        # Bind
        # -----------------------------
        btn_confirm.click(confirm_prompt, inputs=[prompt], outputs=[img, status])
        img.select(on_click, inputs=[mode], outputs=[img, status])
        btn_reset.click(reset_points, outputs=[img, status])
        btn_download_results.click(prepare_results_download, outputs=[results_download, status])
        if args.public_demo:
            btn_load_upload.click(load_uploaded_video, inputs=[upload_video], outputs=[img, preview_img, status])

        # >>> NEW: Bind to correct preview component based on mode
        if is_single_image:
            btn_confirm_mask.click(confirm_mask_and_preview, inputs=None, outputs=[preview_img, status])
            btn_next_obj.click(save_and_next_object, inputs=None, outputs=[img, preview_img, status])
        else:
            btn_confirm_mask.click(confirm_mask_and_preview, inputs=None, outputs=[preview_img, status])
            btn_next_obj.click(save_and_next_object, inputs=None, outputs=[img, preview_img, status])

        # Note: btn_end binding is done later after generation section is defined

        # ============================================================
        # Articulate Object Annotation Mode (appears after clicking End)
        # ============================================================
        
        # Create Articulate Object Manager
        articulate_manager = ArticulateObjectManager(out_dir)
        
        with gr.Column(visible=False) as articulate_section:
            gr.Markdown("## 🎯 Articulate Object Annotation")
            gr.Markdown("Select an object to add joint annotations for IsaacSim")
            
            # >>> NEW: Allow user to specify custom scan path
            with gr.Row():
                scan_path_input = gr.Textbox(
                    value=str(out_dir),
                    label="Scan Directory",
                    placeholder="Enter path to scan for objects...",
                    scale=4
                )
                btn_refresh_objects = gr.Button("🔄 Refresh", scale=1)
                btn_generate_meshes = gr.Button("🎲 Generate Meshes", scale=1)
            
            # >>> NEW: Store choices in State for dynamic updates
            object_choices_state = gr.State(value=[])
            
            with gr.Row():
                # Object list
                object_list = gr.Dataframe(
                    headers=["Name", "Has Joints", "Mesh Count"],
                    value=[],
                    interactive=False,
                    label="Segmented Objects",
                    wrap=True
                )
            
            with gr.Row():
                # Select object dropdown - dynamically updated
                object_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Object to Annotate",
                    interactive=True
                )
                btn_load_object = gr.Button("📂 Load Object for Joint Annotation")

            # 3D Viewer container (will load Three.js HTML)
            viewer_html = gr.HTML(
                value="<div style='text-align: center; padding: 50px; color: #888;'>Select an object above to load 3D viewer</div>",
                sanitize=False,
                label="3D Joint Annotation Viewer"
            )

            # >>> Articulate Tool Section
            with gr.Row():
                btn_load_articulate = gr.Button("🔧 Load Articulate Tool", variant="primary", scale=3)
                btn_open_new_tab = gr.Button("↗️ Open in New Tab", variant="secondary", scale=1, visible=False)
                btn_close_articulate = gr.Button("✖️ Close", variant="stop", scale=1, visible=False)

            articulate_viewer = gr.HTML(
                value="<div style='text-align: center; padding: 50px; color: #888;'>Click 'Load Articulate Tool' to start</div>",
                sanitize=False,
                label="Articulate Tool"
            )

            # Hidden state for current articulate object
            articulate_current_object = gr.State(value=None)

            # Status for articulate mode
            articulate_status = gr.Markdown("🟢 Select an object to begin")

            def load_articulate_viewer(object_name, scan_path):
                """Launch articulate page and return iframe"""
                if not object_name:
                    return (
                        "<div style='text-align: center; padding: 50px; color: #888;'>Please select an object</div>",
                        object_name,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        "❌ No object selected"
                    )

                base_path = scan_path if scan_path else str(out_dir)
                object_dir = Path(base_path) / object_name

                if not object_dir.exists():
                    return (
                        f"<div style='text-align: center; padding: 50px; color: #888;'>Directory not found</div>",
                        object_name,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        "❌ Directory not found"
                    )

                glb_files = list(object_dir.glob("*.glb"))
                if not glb_files:
                    return (
                        f"<div style='text-align: center; padding: 50px; color: #888;'>No GLB files</div>",
                        object_name,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        "❌ No GLB files"
                    )

                missing = _articulate_tool_manager.missing_requirements()
                if missing:
                    message = "<br>".join(missing)
                    return (
                        f"<div style='padding: 20px; color: #8a4b00;'>Articulate tool is not configured:<br>{message}</div>",
                        object_name,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        "Configure the articulate tool app and checkpoint before opening the subpage"
                    )

                print(f"[Articulate Tool] Starting for {object_name}")
                port = _articulate_tool_manager.launch_for_object(object_name, base_path)
                local_url = f"http://127.0.0.1:{port}"
                iframe_url = _articulate_tool_manager.get_iframe_url(object_name)

                print(f"[DEBUG] Waiting for server to start on port {port}...")

                # Wait for server to be ready
                import urllib.request
                max_wait = 60
                for i in range(max_wait):
                    try:
                        urllib.request.urlopen(local_url, timeout=1)
                        print(f"[DEBUG] Server ready after {i+1}s")
                        break
                    except:
                        if i < max_wait - 1:
                            time.sleep(1)
                        else:
                            print(f"[DEBUG] Server not ready after {max_wait}s")

                viewer_html = f"""
                <div style="padding: 10px; background: #f0f0f0; border-radius: 8px; margin-bottom: 10px;">
                    <strong>Port {port}</strong> | <a href="{iframe_url}" target="_blank" style="color: #0066cc;">Open in New Tab (Recommended)</a>
                </div>
                <div style="border: 1px solid #ddd; border-radius: 8px; height: 700px; overflow: hidden;">
                    <iframe
                        src="{iframe_url}"
                        width="100%"
                        height="100%"
                        style="border: none;"
                        sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-modals allow-downloads"
                        referrerpolicy="no-referrer-when-downgrade">
                        <p>Your browser does not support iframes. <a href="{iframe_url}" target="_blank">Click here to open in new tab</a></p>
                    </iframe>
                </div>
                <script>
                console.log('[ArticulateTool] Loading iframe from:', '{iframe_url}');
                // Check if iframe loaded
                setTimeout(() => {{
                    const iframe = document.querySelector('iframe[src="{iframe_url}"]');
                    if (iframe) {{
                        console.log('[ArticulateTool] iframe element found');
                        iframe.onload = () => console.log('[ArticulateTool] iframe loaded successfully');
                        iframe.onerror = (e) => console.error('[ArticulateTool] iframe error:', e);
                    }}
                }}, 100);
                </script>
                """

                return (
                    viewer_html,
                    object_name,
                    gr.update(visible=True),
                    gr.update(visible=True),
                    f"✅ Loading on port {port}"
                )

            def close_articulate_viewer(object_name):
                if object_name:
                    _articulate_tool_manager.stop_object(object_name)
                    print(f"[Articulate Tool] Stopped for {object_name}")
                return (
                    "<div style='text-align: center; padding: 50px; color: #888;'>Tool closed.</div>",
                    None,
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            def open_in_new_tab(object_name):
                """Return JavaScript to open in new tab"""
                if not object_name or object_name not in _articulate_tool_manager.ports:
                    return "❌ No active server"
                port = _articulate_tool_manager.ports[object_name]
                url = _articulate_tool_manager.get_public_url(port)
                # Return HTML with auto-open script
                return f"""
                <script>window.open('{url}', '_blank');</script>
                <div style='text-align: center; padding: 20px; color: #666;'>
                    Opening in new tab... If blocked, <a href="{url}" target="_blank">click here</a>
                </div>
                """

            btn_load_articulate.click(
                load_articulate_viewer,
                inputs=[object_dropdown, scan_path_input],
                outputs=[articulate_viewer, articulate_current_object, btn_open_new_tab, btn_close_articulate, articulate_status]
            )

            btn_open_new_tab.click(
                open_in_new_tab,
                inputs=[articulate_current_object],
                outputs=[articulate_viewer]
            )

            btn_close_articulate.click(
                close_articulate_viewer,
                inputs=[articulate_current_object],
                outputs=[articulate_viewer, articulate_current_object, btn_open_new_tab, btn_close_articulate]
            )
            
            # Hidden state for current object used by the viewer
            current_object_state = gr.State(value=None)
            
            def load_object_for_annotation(object_name, scan_path):
                """Load object meshes for joint annotation"""
                if not object_name:
                    return (
                        "<div style='text-align: center; padding: 50px; color: #888;'>Please select an object</div>",
                        "❌ No object selected"
                    )
                
                # Use the scan path as base for finding meshes
                base_path = scan_path if scan_path else str(out_dir)
                
                # Get mesh files
                meshes = articulate_manager.get_object_meshes(object_name, base_path)
                if not meshes:
                    return (
                        f"<div style='text-align: center; padding: 50px; color: #888;'>No mesh files found for {object_name}</div>",
                        f"❌ No mesh files for {object_name}"
                    )
                
                # Load existing joints if any
                existing_joints = articulate_manager.load_joints(object_name, base_path)
                joint_info = ""
                if existing_joints and existing_joints.get("joints"):
                    joint_info = f"📋 Existing joints: {len(existing_joints['joints'])}"
                
                # Generate HTML with Three.js viewer
                html_template = _get_articulate_viewer_html(Path(base_path), object_name, meshes)
                
                return (
                    html_template,
                    f"✅ Loaded {object_name} with {len(meshes)} mesh(es). {joint_info}"
                )
            
            def save_joints_from_viewer(object_name, joints_data, scan_path):
                """Save joints from 3D viewer to file"""
                if not object_name:
                    return "❌ No object selected"
                
                if not joints_data:
                    return "❌ No joints to save"
                
                base_path = scan_path if scan_path else str(out_dir)
                
                # Save joints
                articulate_manager.save_joints(object_name, joints_data, base_path=base_path)
                
                return f"✅ Saved {len(joints_data)} joint(s) to {object_name}_joints.json"
            
            # Bind events
            def refresh_object_list(scan_path):
                """Refresh the object list from specified directory"""
                if not scan_path:
                    return (
                        gr.update(choices=[]),
                        [],
                        "❌ No scan path specified"
                    )
                
                scan_path_obj = Path(scan_path)
                
                # Check if it's single_mask style (GLB files directly in folder)
                has_direct_glbs = any(f.suffix == '.glb' for f in scan_path_obj.glob("*") if f.is_file())
                
                # Use appropriate scan function
                if has_direct_glbs:
                    # single_mask style: GLB files directly in folder (0.glb, 1.glb, etc.)
                    objects = articulate_manager.scan_single_mask_objects(scan_path)
                else:
                    # multi_mask style: subdirectories with object names (default)
                    objects = articulate_manager.scan_segmented_objects(scan_path)
                
                if not objects:
                    return (
                        gr.update(choices=[]),
                        [],
                        f"❌ No objects found in {scan_path}"
                    )
                
                # Update dropdown choices and table
                choices = [obj["name"] for obj in objects]
                table_data = [
                    [obj["name"], "✅" if obj["has_joints"] else "❌", obj["mesh_count"]]
                    for obj in objects
                ]
                
                return (
                    gr.update(choices=choices),
                    table_data,
                    f"✅ Found {len(objects)} objects in {scan_path}"
                )
            
            def trigger_generate_meshes(scan_path):
                """Generate meshes for all objects"""
                if not scan_path:
                    return "❌ No scan path specified"
                
                result = generate_3d_meshes_for_all_objects(Path(scan_path), SAM3D_CONFIG)
                
                # Refresh list after generation
                objects = articulate_manager.scan_segmented_objects(scan_path)
                choices = [obj["name"] for obj in objects]
                table_data = [
                    [obj["name"], "✅" if obj["has_joints"] else "❌", obj["mesh_count"]]
                    for obj in objects
                ]
                
                # Update UI
                return (
                    result["message"],
                    gr.update(choices=choices),
                    table_data,
                )
            
            # Bind refresh button
            btn_refresh_objects.click(
                refresh_object_list,
                inputs=[scan_path_input],
                outputs=[object_dropdown, object_list, articulate_status]
            )
            
            # Bind generate meshes button
            btn_generate_meshes.click(
                trigger_generate_meshes,
                inputs=[scan_path_input],
                outputs=[articulate_status, object_dropdown, object_list]
            )
            
            btn_load_object.click(
                load_object_for_annotation,
                inputs=[object_dropdown, scan_path_input],
                outputs=[viewer_html, articulate_status]
            )
            
            object_dropdown.change(
                load_object_for_annotation,
                inputs=[object_dropdown, scan_path_input],
                outputs=[viewer_html, articulate_status]
            )

        # ============================================================
        # 3D Generation Section (appears after clicking End)
        # ============================================================
        
        # Global state for tracking generation progress
        generation_progress = gr.State(value={"logs": [], "current": 0, "total": 0, "done": False})
        
        with gr.Column(visible=False) as generation_section:
            gr.Markdown("## 🔄 Generating 3D Meshes")
            gr.Markdown("Please wait while the system generates 3D models from your masks...")
            
            # Progress bar (using HTML for more control)
            progress_html = gr.HTML(
                value="""
                <div style="width: 100%; background: #333; border-radius: 8px; height: 24px; overflow: hidden;">
                    <div id="progress-bar" style="width: 0%; background: linear-gradient(90deg, #4CAF50, #8BC34A); height: 100%; transition: width 0.3s;"></div>
                </div>
                <p id="progress-text" style="text-align: center; color: #888; margin-top: 8px;">Initializing...</p>
                """,
                label="Progress"
            )
            
            # Log display area
            generation_logs = gr.Textbox(
                label="Generation Logs",
                value="Click 'Start Generation' to begin 3D mesh generation...",
                interactive=False,
                lines=15,
                max_lines=20
            )
            
            # Status message
            generation_status = gr.Markdown("🟡 Ready to generate")
            
            # Button to start generation
            btn_start_generation = gr.Button("🚀 Start 3D Generation", variant="primary")
            
            # Button to check progress
            btn_check_progress = gr.Button("🔍 Check Progress / Continue")
            
            # Button to skip and continue (if some meshes are done)
            btn_skip_generation = gr.Button("⏭️ Skip to Articulate (if meshes exist)")
            
            # Hidden continue button (auto-trigger after generation)
            btn_continue_to_articulate = gr.Button("Continue to Articulate", visible=False)
        
        def update_progress_ui(logs, current, total):
            """Update the progress UI"""
            log_text = "\n".join(logs[-20:])  # Show last 20 logs

            progress_html_val = """
            <div style="width: 100%; background: #333; border-radius: 8px; height: 24px; overflow: hidden;">
                <div id="progress-bar" style="width: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); height: 100%; transition: width 0.3s; animation: pulse 1.5s ease-in-out infinite;"></div>
            </div>
            <p id="progress-text" style="text-align: center; color: #888; margin-top: 8px;">Generating...</p>
            """

            return (
                progress_html_val,
                log_text,
                "🔄 Generating..."
            )
        
        def start_3d_generation():
            """Start the 3D generation process in background"""
            import threading
            
            # Use a global variable to track progress
            progress_data = {"logs": [], "current": 0, "total": 0, "done": False}
            
            def progress_callback(step, total, msg):
                progress_data["current"] = step
                progress_data["total"] = total
                progress_data["logs"].append(msg)
                print(f"[PROGRESS] {step}/{total}: {msg}")
            
            def run_generation():
                # First: Generate meshes for all objects
                result = generate_3d_meshes_for_all_objects(
                    out_dir, 
                    SAM3D_CONFIG, 
                    max_frames=20,
                    progress_callback=progress_callback
                )
                
                progress_data["logs"].append(f"=== Mesh Generation Complete: {result.get('message', 'Done')} ===")
                
                # run_inference.py --compose_scene already handles everything (mesh + scale)
                progress_data["logs"].append("=== All Processing Complete! ===")
                progress_data["done"] = True
                print("[GENERATION] All complete!")
            
            # Start generation in background thread
            thread = threading.Thread(target=run_generation)
            thread.daemon = True
            thread.start()
            
            # Return initial state
            return (
                update_progress_ui([], 0, 0)[0],
                "",
                "🔄 Starting 3D generation..."
            )
        
        def check_generation_progress():
            """Poll and update generation progress"""
            # In a real implementation, we'd use a shared state
            # For now, just return a static update that triggers UI refresh
            return (
                update_progress_ui(["Checking progress..."], 1, 10)[0],
                "Generation in progress...",
                "⏳ Still generating..."
            )
        
        def show_articulate_after_generation():
            """Transition to articulate mode after 3D generation is done"""
            # Scan from multi_mask directory (has subdirectories with object names)
            multi_mask_path = out_dir  # This is already case1/multi_mask
            
            # Auto-refresh object list from multi_mask
            objects = articulate_manager.scan_segmented_objects(str(multi_mask_path))
            choices = [obj["name"] for obj in objects]
            table_data = [
                [obj["name"], "✅" if obj["has_joints"] else "❌", obj["mesh_count"]]
                for obj in objects
            ]
            
            return {
                generation_section: gr.update(visible=False),
                articulate_section: gr.update(visible=True),
                status: f"🎯 Articulate Mode - {len(objects)} objects ready",
                scan_path_input: str(multi_mask_path),
                object_dropdown: gr.update(choices=choices),
                object_list: table_data,
            }
        
        # After clicking End, show the generation section first
        def on_end_clicked(status_msg):
            """When End button is clicked, start 3D generation"""
            # First, show the generation section
            return {
                generation_section: gr.update(visible=True),
                status: "🔄 Starting 3D generation...",
            }
        
        btn_end.click(
            on_end_clicked,
            inputs=[status],
            outputs=[generation_section, status]
        )
        
        # Add a timer to periodically update the progress display (every 3 seconds)
        # This creates a polling mechanism to refresh the UI
        import time
        
        def poll_progress():
            """Poll progress - this is called periodically"""
            # Note: In Gradio, we use a client-side polling or manual refresh
            # For simplicity, we'll just show the generation is in progress
            return (
                update_progress_ui(["Generation running in background...", 
                                   "This may take several minutes...",
                                   "Each object needs ~30 seconds for mesh + scale..."], 
                                  1, 10)[0],
                "Generation running in background...\nEach object needs ~30-60 seconds for mesh generation + scale extraction.",
                "🔄 3D Generation in progress..."
            )
        
        # We need a way to auto-continue. Let's add a button to check if done
        btn_check_done = gr.Button("Check if Generation Complete", visible=False)
        
        def attempt_continue():
            """Try to transition to articulate mode, or show still working"""
            # Check if meshes exist
            objects = articulate_manager.scan_segmented_objects(str(out_dir))
            all_have_meshes = all(
                (out_dir / obj["name"] / f"{obj['name']}.glb").exists()
                for obj in objects
            )
            
            if all_have_meshes and len(objects) > 0:
                # Generation complete, show articulate section
                return show_articulate_after_generation()
            else:
                # Still working
                done_count = len([o for o in objects if (out_dir / o["name"] / f"{o['name']}.glb").exists()])
                return {
                    "status": f"⏳ Still generating... ({done_count}/{len(objects)} objects done)"
                }
        
        btn_check_done.click(
            attempt_continue,
            outputs=[generation_section, articulate_section, status, scan_path_input, object_dropdown, object_list]
        )
        
        # Add a "I'm done waiting" button for user to manually check
        with gr.Row(visible=False) as generation_controls:
            btn_manual_check = gr.Button("Check if Done")
        
        def manual_check_done():
            objects = articulate_manager.scan_segmented_objects(str(out_dir))
            done_count = sum(1 for obj in objects if (out_dir / obj["name"] / f"{obj['name']}.glb").exists())
            total_count = len(objects)
            
            if done_count == total_count and total_count > 0:
                return show_articulate_after_generation()
            else:
                return {
                    generation_logs: f"Still generating... {done_count}/{total_count} objects have meshes.\n\nIf some objects already have meshes, you can manually continue to Articulate mode.",
                    generation_status: f"⏳ Progress: {done_count}/{total_count} objects",
                    status: f"⏳ Generating 3D: {done_count}/{total_count} done"
                }
        
        # Update the generation section to include a manual continue option
        def update_generation_section_manual():
            objects = articulate_manager.scan_segmented_objects(str(out_dir))
            done_count = sum(1 for obj in objects if (out_dir / obj["name"] / f"{obj['name']}.glb").exists())
            total_count = len(objects)
            
            # Check if done
            if done_count == total_count and total_count > 0:
                return show_articulate_after_generation()
            
            # Not done yet, show current state
            log_text = f"Generation in progress...\n\nObjects: {total_count}\nCompleted: {done_count}\n\nIf you want to skip waiting and check current status, click 'Check Progress' below.\n\nNote: Each object takes ~30-60 seconds for mesh + scale generation."
            
            return {
                generation_logs: log_text,
                generation_status: f"⏳ {done_count}/{total_count} objects completed",
                status: f"🔄 Generating 3D: {done_count}/{total_count}"
            }
        
        # ============================================================
        # Bind Generation Section Buttons
        # ============================================================
        
        def _start_generation_bg():
            """Start 3D generation in background thread"""
            import threading
            global _GEN_STATE
            
            if _GEN_STATE["running"]:
                # Already running, return current state
                return update_progress_ui(_GEN_STATE["logs"], _GEN_STATE["current"], _GEN_STATE["total"])
            
            # Get object count
            objects = articulate_manager.scan_segmented_objects(str(out_dir))
            obj_count = len(objects)
            
            # Initialize
            _GEN_STATE = {
                "running": True,
                "logs": [f"Starting 3D generation for {obj_count} objects..."],
                "current": 0,
                "total": obj_count,  # mesh + scale
                "done": False
            }
            
            def _progress_cb(step, total, msg):
                global _GEN_STATE
                _GEN_STATE["current"] = step
                _GEN_STATE["total"] = total  # 同步外部传入的 total，避免超标
                _GEN_STATE["logs"].append(msg)
            
            def _run_gen():
                global _GEN_STATE
                try:
                    # Mesh generation (run_inference.py --compose_scene handles everything)
                    result = generate_3d_meshes_for_all_objects(out_dir, SAM3D_CONFIG, max_frames=20, progress_callback=_progress_cb)
                    _GEN_STATE["logs"].append(f"=== Mesh Generation Done: {result.get('message', 'OK')} ===")
                    
                    # run_inference.py --compose_scene already handles scale extraction
                    _GEN_STATE["logs"].append("=== All Complete! ===")
                    _GEN_STATE["done"] = True
                except Exception as e:
                    _GEN_STATE["logs"].append(f"[ERROR] {str(e)}")
                    _GEN_STATE["done"] = True
            
            t = threading.Thread(target=_run_gen)
            t.daemon = True
            t.start()
            
            # Return only 3 values: progress_html, generation_logs, generation_status
            return (
                update_progress_ui([f"Started! {obj_count} objects to process...", 
                                    "Click 'Check Progress' to monitor..."], 0, _GEN_STATE["total"])[0],
                f"Started! {obj_count} objects to process...\nClick 'Check Progress' to monitor...",
                "🚀 Starting 3D generation..."
            )
        
        def _check_progress():
            """Check generation progress and transition if done"""
            global _GEN_STATE
            
            if _GEN_STATE["done"]:
                # Transition to articulate mode
                objects = articulate_manager.scan_segmented_objects(str(out_dir))
                choices = [obj["name"] for obj in objects]
                table_data = [
                    [obj["name"], "✅" if obj["has_joints"] else "❌", obj["mesh_count"]]
                    for obj in objects
                ]
                # Convert logs list to string for markdown component
                log_str = "\n".join(_GEN_STATE["logs"]) if isinstance(_GEN_STATE["logs"], list) else str(_GEN_STATE["logs"])
                return (
                    update_progress_ui(_GEN_STATE["logs"], _GEN_STATE["current"], _GEN_STATE["total"])[0],
                    log_str,
                    "✅ Generation Complete!",
                    gr.update(visible=False),
                    gr.update(visible=True),
                    f"🎯 Articulate Mode - {len(objects)} objects ready",
                    str(out_dir),
                    gr.update(choices=choices),
                    gr.update(value=table_data)
                )
            
            # Convert logs list to string for markdown component
            log_str = "\n".join(_GEN_STATE["logs"]) if isinstance(_GEN_STATE["logs"], list) else str(_GEN_STATE["logs"])
            return update_progress_ui(_GEN_STATE["logs"], _GEN_STATE["current"], _GEN_STATE["total"])[0], log_str, "🔄 Generating...", gr.update(), gr.update(), "", gr.update(), gr.update(), gr.update()
        
        def _skip_to_articulate():
            """Skip to articulate if any meshes exist"""
            objects = articulate_manager.scan_segmented_objects(str(out_dir))
            has_mesh = any((out_dir / o["name"] / f"{o['name']}.glb").exists() for o in objects)
            
            if has_mesh:
                return show_articulate_after_generation()
            return {"generation_logs": "⚠️ No meshes yet. Run generation first or wait.",
                    "generation_status": "❌ No meshes", "status": "❌ Cannot skip"}
        
        # Bind buttons
        btn_start_generation.click(_start_generation_bg, outputs=[progress_html, generation_logs, generation_status])
        btn_check_progress.click(_check_progress, outputs=[progress_html, generation_logs, generation_status, generation_section, articulate_section, status, scan_path_input, object_dropdown, object_list])
        btn_skip_generation.click(_skip_to_articulate, outputs=[generation_logs, generation_status, status, generation_section, articulate_section, status, scan_path_input, object_dropdown, object_list])

        # 自动轮询进度：每 3 秒刷新一次，仅当生成正在运行时有效
        def _auto_poll():
            """Timer callback: 仅在运行中时自动刷新进度"""
            global _GEN_STATE
            if not _GEN_STATE["running"]:
                # 未运行，返回空更新
                return gr.update(), gr.update(), gr.update()
            log_str = "\n".join(_GEN_STATE["logs"]) if isinstance(_GEN_STATE["logs"], list) else str(_GEN_STATE["logs"])
            progress = update_progress_ui(_GEN_STATE["logs"], _GEN_STATE["current"], _GEN_STATE["total"])[0]
            if _GEN_STATE["done"]:
                return progress, log_str, "✅ Generation Complete! Click 'Check Progress' to continue."
            return progress, log_str, "🔄 Generating..."

        gen_timer = gr.Timer(value=3, active=False)
        gen_timer.tick(_auto_poll, outputs=[progress_html, generation_logs, generation_status])
        # 点击 Start 时激活 Timer；生成完成时不需要停，因为 _auto_poll 在 done 后返回静态提示
        btn_start_generation.click(lambda: gr.Timer(active=True), outputs=[gen_timer])

        # ============================================================
        # Cleanup - Stop articulate processes on exit
        # ============================================================
        import atexit

        def cleanup_articulate():
            """Stop all articulate processes on exit"""
            print("[Cleanup] Stopping articulate servers...")
            _articulate_tool_manager.stop_all()
            print("[Cleanup] Done")

        atexit.register(cleanup_articulate)

        # ============================================================
        # Launch Demo
        # ============================================================
        if args.public_demo:
            allowed_candidates = [
                input_path,
                input_path.parent,
                out_dir,
                out_dir.parent,
                download_dir,
                *EXTRA_ALLOWED_ROOTS,
            ]
            blocked_candidates = [
                CHECKPOINT_DIR,
                ROBOSNAP_ROOT / "checkpoints",
                ROBOSNAP_ROOT / ".git",
                ROBOSNAP_ROOT / "data",
                Path.home() / ".ssh",
                *args.blocked_root,
            ]
        else:
            allowed_candidates = [
                ROBOSNAP_ROOT,
                out_dir,
                out_dir.parent,
                out_dir.parent / "single_mask",
                download_dir,
                *EXTRA_ALLOWED_ROOTS,
            ]
            blocked_candidates = [*args.blocked_root]

        allowed_paths = _unique_existing_paths(allowed_candidates)
        blocked_paths = _unique_existing_paths(blocked_candidates)
        auth = _parse_basic_auth(args.auth)

        demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            debug=args.debug,
            show_error=True,
            allowed_paths=allowed_paths,
            blocked_paths=blocked_paths,
            max_file_size=args.max_file_size,
            auth=auth,
            enable_monitoring=False if args.public_demo else None,
            root_path=None,  # Disable root path to allow iframe
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ckpt", default=SAM3_CKPT)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7897")))
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--public-demo", action="store_true")
    parser.add_argument("--max-file-size", default=os.environ.get("GRADIO_MAX_FILE_SIZE"))
    parser.add_argument("--auth", default=os.environ.get("GRADIO_AUTH", ""))
    parser.add_argument("--delete-cache-frequency", type=int, default=int(os.environ.get("GRADIO_DELETE_CACHE_FREQUENCY", "0")))
    parser.add_argument("--delete-cache-age", type=int, default=int(os.environ.get("GRADIO_DELETE_CACHE_AGE", "0")))
    parser.add_argument("--max_frames", type=int, default=20)
    parser.add_argument("--asset-python", default=PY_ASSET)
    parser.add_argument("--asset-dir", dest="asset_dir", default=str(SAM3D_DIR))
    parser.add_argument("--sam3d-dir", dest="asset_dir", help=argparse.SUPPRESS, default=argparse.SUPPRESS)
    parser.add_argument("--asset-config", dest="asset_config", default=SAM3D_CONFIG)
    parser.add_argument("--sam3d-config", dest="asset_config", help=argparse.SUPPRESS, default=argparse.SUPPRESS)
    parser.add_argument("--articulate-python", dest="articulate_python", default=PY_ARTICULATE)
    parser.add_argument("--articulate-app", dest="articulate_app", default=ARTICULATE_APP)
    parser.add_argument("--articulate-ckpt", dest="articulate_ckpt", default=ARTICULATE_CKPT)
    parser.add_argument("--articulate-base-port", dest="articulate_base_port", type=int, default=ARTICULATE_BASE_PORT)
    parser.add_argument("--articulate-public-url-template", dest="articulate_public_url_template", default=os.environ.get("ARTICULATE_PUBLIC_URL_TEMPLATE", os.environ.get("P3SAM_PUBLIC_URL_TEMPLATE")))
    parser.add_argument("--allowed-root", action="append", default=[])
    parser.add_argument("--blocked-root", action="append", default=[])
    args = parser.parse_args()
    apply_runtime_config(args)
    main(args)
