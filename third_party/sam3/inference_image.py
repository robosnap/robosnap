import torch
import numpy as np
import os
import json
import argparse
#################################### For Image###################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor




def parse_prompt_list(prompt_str: str):
    """
    input "a, b, c"
    output ["a", "b", "c"]
    """
    return [p.strip() for p in prompt_str.split(",") if p.strip()]



def save_transparent_mask(
    image: Image.Image,
    masks: torch.Tensor,
    mask_index: int,
    scores: torch.Tensor,
    out_dir="./outputs",
):
    """
    masks: torch.BoolTensor [H, W]
    scores: [N] or [N, 1]
    """
    assert masks.ndim == 4, f"Expected [N,1,H,W], got {masks.shape}"
    if masks.shape[0] == 0:
        print(f"Warning: Can not find mask for the {mask_index} object, skip.")
        return

    best_idx = scores.squeeze(-1).argmax().item()   # 兼容 [N] / [N,1]
    mask = masks[best_idx, 0].detach().cpu().numpy()  # [H, W]
    h, w = mask.shape
    rgb = np.array(image.convert("RGB"))  #[H,W,3]
    alpha = (mask.astype(np.uint8) * 255) #[H,W]
    rgba = np.dstack([rgb, alpha])  # [H, W, 4]


    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{mask_index}.png")
    Image.fromarray(rgba, mode="RGBA").save(out_path)
    print(f"[👌] Saved transparent mask to: {out_path}")
    return out_path


def main(args):
    model = build_sam3_image_model(checkpoint_path=args.checkpoint)
    processor = Sam3Processor(model)

    image = Image.open(args.image).convert("RGB")
    inference_state = processor.set_image(image)

    text_prompts = parse_prompt_list(args.text_prompts) if args.text_prompts else []
    geo_prompts = json.loads(args.geo_prompts) if args.geo_prompts else []

    if args.prompt_mode == 0:        # text only
        num_objects = len(text_prompts)
    elif args.prompt_mode == 1:      # geo only
        num_objects = len(geo_prompts)
    else:                            # text + geo
        num_objects = max(len(text_prompts), len(geo_prompts))

    print(f"Total objects to segment: {num_objects}")

    for mask_index in range(num_objects):
        
        print(f"\n[🔍] Segmenting object {mask_index}")

        processor.reset_all_prompts(inference_state)

        if args.prompt_mode in (0, 2):
            if mask_index < len(text_prompts):
                prompt = text_prompts[mask_index]
                print(f"   ├─ text: {prompt}")
                processor.set_text_prompt(
                    prompt=prompt,
                    state=inference_state
                )

        if args.prompt_mode in (1, 2):
            if mask_index < len(geo_prompts):
                box = geo_prompts[mask_index]   # [cx, cy, w, h]
                print(f"   ├─ geo box: {box}")
                processor.add_geometric_prompt(
                    box=box,
                    label=True,                 # 默认正样本 ✔
                    state=inference_state
                )
        
        if mask_index == 0:
            print("\n=== SAM3 OUTPUT KEYS ===")
            print(inference_state.keys())

            print("\n=== TYPES ===")
            for k, v in inference_state.items():
                if torch.is_tensor(v):
                    print(f"{k:20s}: Tensor {tuple(v.shape)} {v.dtype} {v.device}")
                else:
                    print(f"{k:20s}: {type(v)}")

        masks = inference_state.get("masks", None)
        scores = inference_state.get("scores", None)

        if masks is None or masks.shape[0] == 0:
            print("❗️No mask found, skip")
            continue

        save_transparent_mask(
            image=image,
            masks=masks,
            scores=scores,
            mask_index=mask_index,
            out_dir=args.out_dir
        )

# Get the masks, bounding boxes, and scores
# print("=== SAM3 OUTPUT KEYS ===")
# print(output.keys())
# dict_keys(['original_height', 'original_width', 'backbone_out', 'geometric_prompt', 'masks_logits', 'masks', 'boxes', 'scores'])

# print("\n=== TYPES ===")
# print("masks :", type(output["masks"]))
# print("boxes :", type(output["boxes"]))
# print("scores:", type(output["scores"]))

# print("\n=== SHAPES ===")
# print("masks shape :", output["masks"].shape)
# print("boxes shape :", output["boxes"].shape)
# print("scores shape:", output["scores"].shape)

# print("\n=== DTYPES / DEVICE ===")
# print("masks dtype :", output["masks"].dtype, "device:", output["masks"].device)
# print("boxes dtype :", output["boxes"].dtype, "device:", output["boxes"].device)
# print("scores dtype:", output["scores"].dtype, "device:", output["scores"].device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAM3 image segmentation")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt_mode", 
        type=int, 
        default=0,
        help="0=text, 1=geo, 2=text+geo"
    )
    parser.add_argument(
        "--text_prompts",
        type=str,
        required=True,
        help='Comma separated prompts, e.g. "cat, brown milk carton box"'
    )
    parser.add_argument(
        "--geo_prompts", 
        type=str, 
        default="",
        help="Path to geometric prompt json"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to sam3 checkpoint (.pt)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs",
        help="Base output directory"
    )
    

    args = parser.parse_args()
    main(args)


