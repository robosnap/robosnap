"""
SAM 3D Objects Inference Script
Supports both single-view and multi-view 3D reconstruction

Usage:
    # Multi-view inference (mask_prompt=None, images and masks in same directory, use all images)
    python run_inference.py --input_path ./data/images_and_masks
    
    # Single-view inference (specify a single image name)
    python run_inference.py --input_path ./data/images_and_masks --image_names image1
    
    # Multi-view inference (mask_prompt!=None, images in images/, masks in specified folder)
    python run_inference.py --input_path ./data --mask_prompt stuffed_toy
    
    # Specify multiple image names (can be any filename without extension)
    python run_inference.py --input_path ./data --mask_prompt stuffed_toy --image_names image1,view_a,2
"""
import sys
import os
import argparse
from pathlib import Path
from typing import List, Optional
from loguru import logger
import json
import pickle
import numpy as np
import torch

_BASE_DIR = Path(__file__).resolve().parents[1]
_base_dir_str = str(_BASE_DIR)
if _base_dir_str not in sys.path:
    sys.path.insert(0, _base_dir_str)
from sam3d_objects.inference import Inference
from sam3d_objects.load_images_and_masks import load_images_and_masks_from_path
from sam3d_objects.utils.cross_attention_logger import CrossAttentionLogger


def setup_hf_offline_cache(
    cache_dir: Optional[str] = None,
):
    cache_dir = cache_dir or os.environ.get("HF_HOME")
    if cache_dir:
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def _to_serializable(obj):
    """Make a lightweight JSON-serializable summary for logging/metadata."""
    try:
        import torch
        import numpy as np
    except Exception:
        torch = None
        np = None

    if torch is not None and isinstance(obj, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
            "min": float(obj.min().item()) if obj.numel() > 0 else None,
            "max": float(obj.max().item()) if obj.numel() > 0 else None,
            "mean": float(obj.float().mean().item()) if obj.numel() > 0 else None,
        }
    if np is not None and isinstance(obj, np.ndarray):
        return {
            "type": "np.ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "min": float(np.min(obj)) if obj.size > 0 else None,
            "max": float(np.max(obj)) if obj.size > 0 else None,
            "mean": float(np.mean(obj)) if obj.size > 0 else None,
        }
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    # trimesh / Gaussian / others: store type only
    return {"type": f"{obj.__class__.__module__}.{obj.__class__.__name__}"}


def _safe_torch_save(tensor: "torch.Tensor", path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), str(path))


def _safe_numpy_save(arr: "np.ndarray", path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)


def _try_pickle_dump(obj, path: Path) -> bool:
    """Try best-effort pickle dump; returns True if ok else False."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return True
    except Exception:
        return False


def parse_attention_layers(layers_str: Optional[str]) -> Optional[List[int]]:
    """
    Parse attention layer indices from CLI string.
    """
    if layers_str is None:
        return None
    tokens = [token.strip() for token in layers_str.split(",") if token.strip()]
    if not tokens:
        return None
    indices: List[int] = []
    for token in tokens:
        try:
            indices.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid attention layer index: {token}") from exc
    return indices


def resolve_attention_stages(stage_str: Optional[str]) -> List[str]:
    """
    Normalize stage selection argument.
    """
    if stage_str is None or stage_str.lower() == "both":
        return ["ss", "slat"]
    stage_str = stage_str.lower()
    if stage_str not in {"ss", "slat"}:
        raise ValueError(f"Invalid attention_stage: {stage_str}")
    return [stage_str]


def get_output_dir(
    input_path: Path, 
    mask_prompt: Optional[str] = None, 
    image_names: Optional[List[str]] = None,
    is_single_view: bool = False
) -> Path:
    """
    Create output directory based on input path and parameters
    
    Args:
        input_path: Input path
        mask_prompt: Mask folder name (if using separated directory structure)
        image_names: List of image names
        is_single_view: Whether it's single-view inference
    
    Returns:
        output_dir: Object folder path (same folder as input PNGs)
    """
    output_dir = input_path if input_path.is_dir() else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    return output_dir

def run_inference(
    input_path: Path,
    seed: int = 42,
    stage1_steps: int = 50,
    stage2_steps: int = 25,
    decode_formats: List[str] = None,
    model_tag: str = "hf",
    config_path: Optional[str] = None,
    save_attention: bool = False,
    attention_stage: Optional[str] = None,
    attention_layers: Optional[List[int]] = None,
    save_coords: bool = False,
    inference: Optional[Inference] = None,
):
    """
    Run inference
    
    Args:
        input_path: Input path
        seed: Random seed
        stage1_steps: Stage 1 inference steps
        stage2_steps: Stage 2 inference steps
        decode_formats: List of decode formats
        model_tag: Model tag
        config_path: Pipeline config path override
        save_attention: Whether to record cross-attention weights
        attention_stage: Stage selector ('ss', 'slat', or 'both')
        attention_layers: Layer indices to record (supports negative indices)
        save_coords: Whether to save 3D spatial coordinates in SLAT attention files
    """
    setup_hf_offline_cache()
    if inference is None:
        # Initialize inference only if not provided (backward compatible)
        if config_path:
            config_path = str(config_path)
        else:
            config_path = f"checkpoints/{model_tag}/pipeline.yaml"
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Model config file not found: {config_path}")
        
        logger.info(f"Loading model: {config_path}")
        inference = Inference(config_path, compile=False)
    else:
        logger.info("Using provided inference instance")
    
    if hasattr(inference._pipeline, 'rendering_engine'):
        if inference._pipeline.rendering_engine != "pytorch3d":
            logger.warning(f"Rendering engine is set to {inference._pipeline.rendering_engine}, changing to pytorch3d")
            inference._pipeline.rendering_engine = "pytorch3d"
    
    logger.info(f"Loading data: {input_path}")
    logger.info("Input format: RGBA-only masks in a single folder")
    
    view_images, view_masks = load_images_and_masks_from_path(
        input_path=input_path,
    )
    
    num_views = len(view_images)
    logger.info(f"Successfully loaded {num_views} views")
    
    is_single_view = num_views == 1
    output_dir = get_output_dir(input_path.parent, None, None, is_single_view)
    
    # 将日志写入输出目录中的 inference.log，方便后续分析
    log_file = output_dir / "inference.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
    )
    decode_formats = decode_formats or ["gaussian", "mesh"]

    attention_logger: Optional[CrossAttentionLogger] = None
    if save_attention:
        stages = resolve_attention_stages(attention_stage)
        attention_dir = output_dir / "attention"
        attention_logger = CrossAttentionLogger(
            attention_dir,
            enabled_stages=stages,
            layer_indices=attention_layers,
            save_coords=save_coords,
        )
        attention_logger.attach_to_pipeline(inference._pipeline)
        logger.info(
            f"Cross-attention logging enabled → stages={stages}, layers={attention_layers or 'default (-1)'}, "
            f"save_coords={save_coords}"
        )

    if is_single_view:
        logger.info("Single-view inference mode")
        image = view_images[0]
        mask = view_masks[0] if view_masks else None
        result = inference._pipeline.run(
            image,
            mask,
            seed=seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=True,
            stage1_inference_steps=stage1_steps,
            stage2_inference_steps=stage2_steps,
            decode_formats=decode_formats,
            attention_logger=attention_logger,
        )
    else:
        logger.info("Multi-view inference mode")
        result = inference._pipeline.run_multi_view(
            view_images=view_images,
            view_masks=view_masks,
            seed=seed,
            mode="multidiffusion",
            stage1_inference_steps=stage1_steps,
            stage2_inference_steps=stage2_steps,
            decode_formats=decode_formats,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=True,
            attention_logger=attention_logger,
        )
    
    # ----------------------------
    # DEBUG result
    # ----------------------------
    logger.debug("-" * 60)
    logger.debug("Result key_dict details")
    logger.debug(f"Keys: {list(result.keys())}")

    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            logger.debug(f"{key}: torch.Tensor shape={tuple(value.shape)} dtype={value.dtype}")
        elif isinstance(value, np.ndarray):
            logger.debug(f"{key}: np.ndarray shape={value.shape} dtype={value.dtype}")
        else:
            logger.debug(f"{key}: {type(value).__name__}")
    logger.debug("=" * 60)
    
    
    saved_files = []
    
    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Generated coordinates: {result['coords'].shape[0] if 'coords' in result else 'N/A'}")
    print(f"{'='*60}")
    
    object_name = input_path.parent.name
    
    print("Result keys: ", result.keys())
    print("Type of all result keys: ", {key: type(result[key]) for key in result.keys()})

    print(f"\n{'-'*60}")
    print(f"All output files saved to: {output_dir}")
    print(f"Saved files: {', '.join(saved_files)}")
    print(f"{'-'*60}")
    
    if attention_logger is not None:
        attention_logger.close()
    
    print(f"\nFile descriptions:")
    print(f"- PLY file: Gaussian Splatting format with position and color information")
    print(f"  * Recommended to use specialized Gaussian Splatting viewers")
    print(f"- GLB file: Complete 3D mesh model, can be viewed in Blender, Three.js, etc.")

    return result


# --------------------------- Scene Compose (Mesh) ---------------------------
# Similar to make_scene in inference.py, but operates on mesh instead of gaussian

def make_scene_mesh(*outputs):
    """
    Compose multiple object meshes into a scene.
    Similar to make_scene in inference.py, but for mesh.
    
    Args:
        *outputs: dicts with keys 'glb' (trimesh), 'rotation', 'translation', 'scale'
    
    Returns:
        scene_mesh: merged trimesh
    """
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation as R
    
    def apply_pose_to_mesh(mesh, quat, trans, scale):

        import torch

        mesh = mesh.copy()
        V = mesh.vertices.astype(np.float64)

        if isinstance(quat, torch.Tensor):
            quat = quat.detach().cpu().numpy()
        if isinstance(trans, torch.Tensor):
            trans = trans.detach().cpu().numpy()
        if isinstance(scale, torch.Tensor):
            scale = scale.detach().cpu().numpy()

        quat = quat.flatten()
        trans = trans.flatten()
        scale = scale.flatten()

        # Y-up → Z-up
        R_YUP_TO_ZUP = np.array([
            [1,0,0],
            [0,0,-1],
            [0,1,0]
        ])

        # Z-up → Y-up
        R_ZUP_TO_YUP = np.array([
            [1,0,0],
            [0,0,1],
            [0,-1,0]
        ])

        V = V @ R_YUP_TO_ZUP

        R_mat = R.from_quat(quat).as_matrix()

        V = (V * scale) @ R_mat.T + trans

        V = V @ R_ZUP_TO_YUP

        mesh.vertices = V

        return mesh
    
    all_meshes = []
    for output in outputs:
        mesh = output['glb']
        quat = output['rotation']  # xyzw
        trans = output['translation']
        scale = output['scale']
        
        # Apply pose to mesh
        mesh_scene = apply_pose_to_mesh(mesh, quat, trans, scale)
        all_meshes.append(mesh_scene)
    
    # Merge all meshes
    if not all_meshes:
        return None
    scene = trimesh.Scene()

    for i, mesh in enumerate(all_meshes):
        scene.add_geometry(mesh, node_name=f"obj_{i}")
    return scene


def compose_scene_from_results(results, output_path):
    """
    Compose scene from multiple inference results.
    
    Args:
        results: list of dicts with 'glb', 'rotation', 'translation', 'scale'
        output_path: path to save composed scene GLB
    """
    if len(results) == 0:
        logger.warning("No results to compose")
        return None
    
    scene = make_scene_mesh(*results)
    if scene is not None:
        scene.export(str(output_path))
        logger.info(f"✓ Composed scene saved to: {output_path}")
    return scene


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3D Objects Inference Script - Supports single-view and multi-view 3D reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-view inference (mask_prompt=None, images and masks in same directory, use all images)
  python run_inference.py --input_path ./data/images_and_masks
  
  # Single-view inference (specify a single image name)
  python run_inference.py --input_path ./data/images_and_masks --image_names image1
  
  # Multi-view inference (mask_prompt!=None, images in images/, masks in specified folder)
  python run_inference.py --input_path ./data --mask_prompt stuffed_toy
  
  # Specify multiple image names (can be any filename without extension)
  python run_inference.py --input_path ./data --mask_prompt stuffed_toy --image_names image1,view_a,2
        """
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input path. If mask_prompt=None, images and masks are in this directory; "
             "if mask_prompt!=None, images are in input_path/images/, masks in input_path/{mask_prompt}/"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--stage1_steps",
        type=int,
        default=50,
        help="Stage 1 inference steps (default: 50)"
    )
    parser.add_argument(
        "--stage2_steps",
        type=int,
        default=25,
        help="Stage 2 inference steps (default: 25)"
    )
    
    parser.add_argument(
        "--decode_formats",
        type=str,
        default="gaussian,mesh",
        help="Decode formats, comma-separated, e.g., 'gaussian,mesh' or 'gaussian' (default: gaussian,mesh)"
    )
    
    parser.add_argument(
        "--model_tag",
        type=str,
        default="hf",
        help="Model tag (default: hf)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override pipeline config path. If set, --model_tag is ignored.",
    )
    parser.add_argument(
        "--save_attention",
        action="store_true",
        help="Enable saving cross-attention weights for analysis",
    )
    parser.add_argument(
        "--attention_stage",
        type=str,
        default="both",
        choices=["ss", "slat", "both"],
        help="Which stage(s) to record: ss, slat, or both (default)",
    )
    parser.add_argument(
        "--attention_layers",
        type=str,
        default="-1",
        help="Comma-separated layer indices to record (supports negative indices, default: -1)",
    )
    parser.add_argument(
        "--save_coords",
        action="store_true",
        help="Save 3D spatial coordinates in SLAT attention files (default: False)",
    )
    parser.add_argument(
        "--compose_scene",
        action="store_true",
        help="Compose all objects into a single scene GLB after inference (for multi-object scenes)",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    decode_formats = [fmt.strip() for fmt in args.decode_formats.split(",") if fmt.strip()]
    if not decode_formats:
        decode_formats = ["gaussian", "mesh"]
    
    try:
        # Initialize inference once before processing any objects
        setup_hf_offline_cache()
        if args.config:
            config_path = str(args.config)
        else:
            config_path = f"checkpoints/{args.model_tag}/pipeline.yaml"
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Model config file not found: {config_path}")
        
        logger.info(f"Loading model: {config_path}")
        inference = Inference(config_path, compile=False)
        
        if hasattr(inference._pipeline, 'rendering_engine'):
            if inference._pipeline.rendering_engine != "pytorch3d":
                logger.warning(f"Rendering engine is set to {inference._pipeline.rendering_engine}, changing to pytorch3d")
                inference._pipeline.rendering_engine = "pytorch3d"
        
        # ---------------------------------------------------------
        # custom object detection logic
        # case1/
        #   object_name/
        #       top*_mask/*.png
        # ---------------------------------------------------------

        if input_path.is_dir():

            subdirs = []

            for obj_dir in sorted(input_path.iterdir()):

                if not obj_dir.is_dir():
                    continue

                object_name = obj_dir.name

                # skip background
                if object_name == "background":
                    logger.info("Skipping background folder")
                    continue

                # find top*_mask
                mask_dir = None
                for child in obj_dir.iterdir():
                    if child.is_dir() and child.name.startswith("top") and child.name.endswith("_mask"):
                        mask_dir = child
                        break

                if mask_dir is None:
                    logger.warning(f"No top*_mask folder in {obj_dir}")
                    continue

                subdirs.append(mask_dir)

            if subdirs:

                logger.info(f"Detected {len(subdirs)} objects")

                base_dir = input_path

                for subdir in subdirs:

                    object_name = subdir.parent.name

                    logger.info(f"Running inference on: {subdir}")

                    result = run_inference(
                        input_path=subdir,
                        seed=args.seed,
                        stage1_steps=args.stage1_steps,
                        stage2_steps=args.stage2_steps,
                        decode_formats=decode_formats,
                        model_tag=args.model_tag,
                        config_path=args.config,
                        save_attention=args.save_attention,
                        attention_stage=args.attention_stage,
                        attention_layers=parse_attention_layers(args.attention_layers),
                        save_coords=args.save_coords,
                        inference=inference,
                    )

                    if 'glb' not in result or result['glb'] is None:
                        logger.warning(f"No GLB generated for {object_name}")
                        continue

                    # -------------------------
                    # Save GLB to multi_mask/object_name/
                    # -------------------------
                    obj_dir = base_dir / object_name
                    obj_glb_path = obj_dir / f"{object_name}.glb"
                    result['glb'].export(str(obj_glb_path))

                    # -------------------------
                    # Save PLY (Gaussian) to multi_mask/object_name/
                    # -------------------------
                    obj_ply_path = obj_dir / f"{object_name}.ply"

                    if 'gs' in result:
                        result['gs'].save_ply(str(obj_ply_path))
                    elif 'gaussian' in result and isinstance(result['gaussian'], list):
                        result['gaussian'][0].save_ply(str(obj_ply_path))
                    else:
                        logger.warning(f"No gaussian found for {object_name}")

                    logger.info(f"Saved {obj_glb_path}")

                return

        run_inference(
            input_path=input_path,
            seed=args.seed,
            stage1_steps=args.stage1_steps,
            stage2_steps=args.stage2_steps,
            decode_formats=decode_formats,
            model_tag=args.model_tag,
            config_path=args.config,
            save_attention=args.save_attention,
            attention_stage=args.attention_stage,
            attention_layers=parse_attention_layers(args.attention_layers),
            save_coords=args.save_coords,
            inference=inference,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
