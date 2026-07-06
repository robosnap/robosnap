import os 
import sys 
from pathlib import Path


def _clear_socks_proxy_env():
    if os.environ.get("ROBOSNAP_KEEP_PROXY") == "1":
        return
    for proxy_var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        value = os.environ.get(proxy_var, "")
        if value.lower().startswith("socks"):
            os.environ.pop(proxy_var, None)


_clear_socks_proxy_env()
import torch
import torch.nn as nn
import numpy as np
import argparse
import viser
import trimesh
from sklearn.decomposition import PCA
import time
import http
from websockets.http11 import Response as WsResponse
from websockets.datastructures import Headers

# ============================================================
# CORS Support for iframe embedding
# ============================================================
_orig_response_init = WsResponse.__init__

def _patched_response_init(self, status, phrase, headers=None, body=b""):
    if headers is None:
        headers = Headers()
    elif isinstance(headers, dict):
        new_headers = Headers()
        for k, v in headers.items():
            new_headers[k] = v
        headers = new_headers

    # Add CORS headers for iframe embedding
    headers["Access-Control-Allow-Origin"] = "*"
    headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD"
    headers["Access-Control-Allow-Headers"] = "*"
    headers["Access-Control-Allow-Credentials"] = "true"

    _orig_response_init(self, status, phrase, headers, body)

WsResponse.__init__ = _patched_response_init

_P3SAM_ROOT = Path(__file__).resolve().parents[1]
if str(_P3SAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_P3SAM_ROOT))
from model import build_P3SAM, load_state_dict

class P3SAM(nn.Module):
    def __init__(self):
        super().__init__()
        build_P3SAM(self)
    
    def load_state_dict(self, 
                        ckpt_path=None, 
                        state_dict=None, 
                        strict=True, 
                        assign=False, 
                        ignore_seg_mlp=False, 
                        ignore_seg_s2_mlp=False, 
                        ignore_iou_mlp=False):
        load_state_dict(self, 
                        ckpt_path=ckpt_path, 
                        state_dict=state_dict, 
                        strict=strict, 
                        assign=assign, 
                        ignore_seg_mlp=ignore_seg_mlp, 
                        ignore_seg_s2_mlp=ignore_seg_s2_mlp, 
                        ignore_iou_mlp=ignore_iou_mlp)

POINT_COLOR = np.array([255, 153, 153])
POINT_SIZE = 0.001
PROMPT_COLOR = np.array([0, 255, 0])
NEG_PROMPT_COLOR = np.array([255, 60, 60])
MASK_COLOR = np.array([0, 0, 255])

def normalize_pc(pc):
    '''
    pc: (N, 3)
    '''
    max_, min_ = np.max(pc, axis=0), np.min(pc, axis=0)
    center = (max_ + min_) / 2
    scale = (max_ - min_) / 2
    scale = np.max(np.abs(scale))
    pc = (pc - center) / (scale + 1e-10)
    return pc

@torch.no_grad()
def get_feat(model, points, normals):
    data_dict = {
        "coord": points,
        "normal": normals,
        "color": np.ones_like(points),
        "batch": np.zeros(points.shape[0], dtype=np.int64)
    }
    data_dict = model.transform(data_dict)
    for k in data_dict:
        if isinstance(data_dict[k], torch.Tensor):
            data_dict[k] = data_dict[k].cuda()
    point = model.sonata(data_dict)
    while "pooling_parent" in point.keys():
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    feat = point.feat # [M, 1232]
    feat = model.mlp(feat) # [M, 512]
    feat = feat[point.inverse] # [N, 512]
    feats = feat
    return feats

@torch.no_grad()
def get_single_prompt_mask(model, feats, points, point_prompt):
    point_num = points.shape[0]
    points = torch.from_numpy(points).float().cuda()   # [N, 3]
    prompt_coord = torch.from_numpy(point_prompt).float().cuda().unsqueeze(0)  # [1, 3]
    prompt_coord = prompt_coord.repeat(point_num, 1) # [N, 3]
    feats_seg = torch.cat([feats, points, prompt_coord], dim=-1) # [N, 512+3+3]

    # 预测mask stage-1
    pred_mask_1 = model.seg_mlp_1(feats_seg).squeeze(-1) # [N]
    pred_mask_2 = model.seg_mlp_2(feats_seg).squeeze(-1) # [N]
    pred_mask_3 = model.seg_mlp_3(feats_seg).squeeze(-1) # [N]
    pred_mask = torch.stack([pred_mask_1, pred_mask_2, pred_mask_3], dim=-1) # [N, 3]

    # 预测mask stage-2
    feats_seg_2 = torch.cat([feats_seg, pred_mask], dim=-1) # [N, 512+3+3+3]
    feats_seg_global = model.seg_s2_mlp_g(feats_seg_2) # [N, 512]
    feats_seg_global = torch.max(feats_seg_global, dim=0).values # [512]
    feats_seg_global = feats_seg_global.unsqueeze(0).repeat(point_num, 1) # [N, 512]
    feats_seg_3 = torch.cat([feats_seg_global, feats_seg_2], dim=-1) # [N, 512+3+3+3+512]
    pred_mask_s2_1 = model.seg_s2_mlp_1(feats_seg_3).squeeze(-1) # [N]
    pred_mask_s2_2 = model.seg_s2_mlp_2(feats_seg_3).squeeze(-1) # [N]
    pred_mask_s2_3 = model.seg_s2_mlp_3(feats_seg_3).squeeze(-1) # [N]
    pred_mask_s2 = torch.stack([pred_mask_s2_1, pred_mask_s2_2, pred_mask_s2_3], dim=-1) # [N, 3]


    mask_1 = torch.sigmoid(pred_mask_s2_1)
    mask_2 = torch.sigmoid(pred_mask_s2_2)
    mask_3 = torch.sigmoid(pred_mask_s2_3)

    mask_1 = mask_1.detach().cpu().numpy() > 0.5
    mask_2 = mask_2.detach().cpu().numpy() > 0.5
    mask_3 = mask_3.detach().cpu().numpy() > 0.5

    print(feats_seg.shape, pred_mask.shape)
    feats_iou = torch.cat([feats_seg_global, feats_seg, pred_mask_s2], dim=-1) # [N, 512+3+3+3+512]
    feats_iou = model.iou_mlp(feats_iou) # [N, 512]
    feats_iou = torch.max(feats_iou, dim=0).values # [512]
    pred_iou = model.iou_mlp_out(feats_iou) # [3]
    pred_iou = torch.sigmoid(pred_iou) # [3]
    org_iou = pred_iou.detach().cpu().numpy() # [3]
    org_iou_1 = org_iou[0].item()
    org_iou_2 = org_iou[1].item()
    org_iou_3 = org_iou[2].item()
    pred_iou_1 = org_iou_1
    pred_iou_2 = org_iou_2
    pred_iou_3 = org_iou_3

    return mask_1, mask_2, mask_3, pred_iou_1, pred_iou_2, pred_iou_3, org_iou_1, org_iou_2, org_iou_3


@torch.no_grad()
def get_mask(model, feats, points, point_prompts, prompt_labels=None, return_components=False):
    point_prompts = np.asarray(point_prompts, dtype=np.float32)
    if point_prompts.ndim == 1:
        point_prompts = point_prompts.reshape(1, 3)
    if point_prompts.size == 0:
        empty_mask = np.zeros(points.shape[0], dtype=bool)
        result = (empty_mask, empty_mask, empty_mask, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if return_components:
            empty_masks = [empty_mask.copy() for _ in range(3)]
            return result + (empty_masks, empty_masks)
        return result

    if prompt_labels is None:
        prompt_labels = np.ones(len(point_prompts), dtype=np.int8)
    else:
        prompt_labels = np.asarray(prompt_labels, dtype=np.int8)

    pos_masks = [np.zeros(points.shape[0], dtype=bool) for _ in range(3)]
    neg_masks = [np.zeros(points.shape[0], dtype=bool) for _ in range(3)]
    pos_ious = []
    org_pos_ious = []

    # Run prompts one by one to avoid multiplying CUDA memory by prompt count.
    for prompt, label in zip(point_prompts, prompt_labels):
        (
            mask_1,
            mask_2,
            mask_3,
            pred_iou_1,
            pred_iou_2,
            pred_iou_3,
            org_iou_1,
            org_iou_2,
            org_iou_3,
        ) = get_single_prompt_mask(model, feats, points, prompt)
        prompt_masks = [mask_1, mask_2, mask_3]

        if label >= 0:
            for i in range(3):
                pos_masks[i] |= prompt_masks[i]
            pos_ious.append([pred_iou_1, pred_iou_2, pred_iou_3])
            org_pos_ious.append([org_iou_1, org_iou_2, org_iou_3])
        else:
            for i in range(3):
                neg_masks[i] |= prompt_masks[i]

    mask_1 = pos_masks[0] & ~neg_masks[0]
    mask_2 = pos_masks[1] & ~neg_masks[1]
    mask_3 = pos_masks[2] & ~neg_masks[2]

    if pos_ious:
        pred_ious = np.asarray(pos_ious, dtype=np.float32).mean(axis=0)
        org_ious = np.asarray(org_pos_ious, dtype=np.float32).mean(axis=0)
    else:
        pred_ious = np.zeros(3, dtype=np.float32)
        org_ious = np.zeros(3, dtype=np.float32)

    result = (
        mask_1,
        mask_2,
        mask_3,
        pred_ious[0].item(),
        pred_ious[1].item(),
        pred_ious[2].item(),
        org_ious[0].item(),
        org_ious[1].item(),
        org_ious[2].item(),
    )
    if return_components:
        return result + ([m.copy() for m in pos_masks], [m.copy() for m in neg_masks])
    return result

def mask2color(mask):
    point_num = mask.shape[0]
    colors = np.expand_dims(POINT_COLOR, axis=0)
    colors = np.tile(colors, (point_num, 1))
    colors[mask] = MASK_COLOR
    return colors


def bbox_face_ids_from_point_mask(mesh, face_idx, point_mask, bbox_padding=0.01):
    masked_face_count = np.bincount(face_idx[point_mask], minlength=len(mesh.faces))
    nonmasked_face_count = np.bincount(face_idx[~point_mask], minlength=len(mesh.faces))
    face_mask_ratio = masked_face_count / (masked_face_count + nonmasked_face_count + 1e-8)
    seed_face_ids = np.where(face_mask_ratio > 0.5)[0]
    if len(seed_face_ids) == 0:
        return seed_face_ids, seed_face_ids

    orig_v = mesh.vertices
    orig_f = mesh.faces
    masked_face_verts = orig_v[orig_f[seed_face_ids].reshape(-1)]
    bbox_min = masked_face_verts.min(axis=0)
    bbox_max = masked_face_verts.max(axis=0)
    padding = (bbox_max - bbox_min) * bbox_padding
    bbox_min -= padding
    bbox_max += padding

    vertex_in_bbox = (
        (orig_v[:, 0] >= bbox_min[0]) & (orig_v[:, 0] <= bbox_max[0]) &
        (orig_v[:, 1] >= bbox_min[1]) & (orig_v[:, 1] <= bbox_max[1]) &
        (orig_v[:, 2] >= bbox_min[2]) & (orig_v[:, 2] <= bbox_max[2])
    )
    f0_in = vertex_in_bbox[orig_f[:, 0]]
    f1_in = vertex_in_bbox[orig_f[:, 1]]
    f2_in = vertex_in_bbox[orig_f[:, 2]]
    face_ids = np.where(f0_in & f1_in & f2_in)[0]
    return face_ids, seed_face_ids


def build_submesh_from_face_ids(mesh, face_ids):
    """Extract selected original faces without mesh simplification."""
    orig_v = mesh.vertices
    orig_f = mesh.faces
    face_ids = np.asarray(face_ids, dtype=np.int64)
    vertex_ids = np.unique(orig_f[face_ids].reshape(-1))

    v_id_map = np.full(len(orig_v), -1, dtype=np.int64)
    v_id_map[vertex_ids] = np.arange(len(vertex_ids))

    sel_v = orig_v[vertex_ids]
    sel_f = v_id_map[orig_f[face_ids]]

    if (
        hasattr(mesh.visual, "face_colors")
        and mesh.visual.face_colors is not None
        and len(mesh.visual.face_colors) == len(mesh.faces)
    ):
        face_colors = mesh.visual.face_colors[face_ids].copy()
    else:
        face_colors = np.full((len(face_ids), 4), 200, dtype=np.uint8)
    face_colors[:, 3] = 255

    mesh_save = trimesh.Trimesh(
        vertices=sel_v,
        faces=sel_f,
        process=False,
    )
    mesh_save.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.full((len(mesh_save.vertices), 4), 255, dtype=np.uint8)
    )
    mesh_save.visual.face_colors = face_colors
    return mesh_save, vertex_ids

def main(args):
    # load model
    print("加载模型")
    model = P3SAM()
    model.load_state_dict(args.ckpt_path)
    model.eval()
    model.cuda()  
    print("模型加载完成")

    print("加载数据列表")
    # Scan for .glb files: first scan current directory, then subdirectories
    _all_glbs = {}

    # 1. Scan current directory for .glb files
    _current_glbs = [f for f in os.listdir(args.data_dir) if f.endswith('.glb')]
    for _glb_file in _current_glbs:
        _name = _glb_file.replace('.glb', '')
        _all_glbs[_name] = os.path.join(args.data_dir, _glb_file)
        print(f"  Found in current dir: {_name} -> {_all_glbs[_name]}")

    # 2. Scan subdirectories for .glb files
    for _sub in sorted(os.listdir(args.data_dir)):
        _sub_path = os.path.join(args.data_dir, _sub)
        if os.path.isdir(_sub_path):
            _glb_files = [f for f in os.listdir(_sub_path) if f.endswith('.glb')]
            if _glb_files:
                _all_glbs[_sub] = os.path.join(_sub_path, _glb_files[0])
                print(f"  Found in subdir: {_sub} -> {_all_glbs[_sub]}")

    object_names = sorted(_all_glbs.keys())
    print(f"共发现{len(object_names)}个物体: {object_names}")

    cur_glb_path = [None]
    if object_names:
        cur_glb_path[0] = _all_glbs[object_names[0]]

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+y")

    points = [None]
    points_handle = [None]
    colors_pca = [None]
    feats = [None]
    show_colors = [None]
    point_prompts = [[]]
    point_prompt_labels = [[]]
    mask_res = [None, None, None]
    positive_mask_res = [None, None, None]
    negative_mask_res = [None, None, None]
    iou_res = [None, None, None]
    iou_org = [None, None, None]
    best = [None]
    mesh_obj = [None]    # 原始 mesh (trimesh)
    face_idx_arr = [None]  # 采样点对应的 face_idx (N,)
    save_counter = [0]  # 每次保存自增，避免编号混乱
    save_ui_handles = [None]  # 保存当前 save UI 的 handle 列表

    def remove_save_ui():
        """移除当前 save UI 的所有元素"""
        if save_ui_handles[0] is not None:
            for _h in save_ui_handles[0]:
                if _h is not None:
                    _h.remove()
            save_ui_handles[0] = None

    def remove_point_prompt():
        for i in range(len(point_prompts[0])):
            server.scene.remove_by_name(f"/prompt_sphere_{i}")
        point_prompts[0] = []
        point_prompt_labels[0] = []

    def clear_state():
        mask_res[0] = None
        mask_res[1] = None
        mask_res[2] = None
        positive_mask_res[0] = None
        positive_mask_res[1] = None
        positive_mask_res[2] = None
        negative_mask_res[0] = None
        negative_mask_res[1] = None
        negative_mask_res[2] = None
        iou_res[0] = None
        iou_res[1] = None
        iou_res[2] = None
        iou_org[0] = None
        iou_org[1] = None
        iou_org[2] = None
        best[0] = None
        remove_point_prompt()

    def load_pc(glb_path=None, use_normal=True, noise_std=0):
        clear_state()

        if glb_path is None:
            glb_path = cur_glb_path[0]
        cur_glb_path[0] = glb_path   

        print(f"加载数据: {glb_path}")
        if glb_path.endswith('.glb') or glb_path.endswith('.obj'):
            mesh = trimesh.load(glb_path, force='mesh', process=False)
            # 面积加权采样：face_weight = 面积比例 → 大面采多、小面采少
            # 10万点 / 61.4万面 → 期望覆盖 ~63% 的面
            # 50万点 → 期望覆盖 ~98% 的面（默认已改为 500k）
            areas = mesh.area_faces
            n_faces = len(mesh.faces)
            face_weight = (areas / areas.sum()).astype(np.float64)
            _points, _face_idx = trimesh.sample.sample_surface(
                mesh, args.point_num, face_weight=face_weight
            )
            _points = normalize_pc(_points)
            _points = _points + np.random.normal(0, 1, size=_points.shape) * noise_std
            normals = mesh.face_normals[_face_idx]
            if not use_normal or args.no_normal:
                normals = normals * 0
        else:
            raise ValueError(f"Unsupported file type: {glb_path}")

        show_color = np.array([POINT_COLOR])
        _show_colors = np.tile(show_color, (_points.shape[0], 1))

        print("预处理特征")
        _feats = get_feat(model, _points, normals)

        print("PCA获取特征颜色")
        feat_save = _feats.float().detach().cpu().numpy()
        data_scaled = feat_save / np.linalg.norm(feat_save, axis=-1, keepdims=True)
        pca = PCA(n_components=3)
        data_reduced = pca.fit_transform(data_scaled)
        data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
        _colors_pca = (data_reduced * 255).astype(np.uint8)

        # add point cloud
        _points_handle = server.scene.add_point_cloud(
            name="/point_cloud",
            points=_points,
            colors=_show_colors,
            point_size=POINT_SIZE,
        )
        points[0] = _points
        points_handle[0] = _points_handle
        colors_pca[0] = _colors_pca
        feats[0] = _feats
        show_colors[0] = _show_colors
        mesh_obj[0] = mesh
        face_idx_arr[0] = _face_idx
        print(f"加载数据完成: {_points.shape[0]} 点, {n_faces} 面, 采样覆盖率 ~{min(args.point_num * 100 / max(n_faces, 1), 100):.0f}%")

    load_pc()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:

        data_list_handle = client.gui.add_dropdown(
            "Object", object_names,
            initial_value=object_names[0] if object_names else ""
        )

        click_button_handle = client.gui.add_button(
            "Add Point Prompt", icon=viser.Icon.POINTER
        )

        prompt_mode_handle = client.gui.add_dropdown(
            "Prompt Mode", ["Include", "Exclude"], initial_value="Include"
        )

        clear_button_handle = client.gui.add_button(
            "Clear Point Prompt", icon=viser.Icon.X
        )

        save_button_handle = client.gui.add_button(
            "Save Current Mask", icon=viser.Icon.DOWNLOAD
        )

        markdown_handle = client.gui.add_markdown(
            "IOU: 1: 0.000, 2: 0.000, 3: 0.000"
        )

        drop_down_handle = client.gui.add_dropdown(
            "Segmentation", ["Mask-1", "Mask-2", "Mask-3"]
        )

        checkbox_handle = client.gui.add_checkbox(
            "Show Feature", initial_value=False
        )

        checkbox_handle_2 = client.gui.add_checkbox(
            "use normal", initial_value=True
        )

        slider_handle = client.gui.add_slider(
            "Point Size",
            min=0.00025,
            max=0.005,
            step=0.00025,
            initial_value=0.001,
        )

        slider_handle_2 = client.gui.add_slider(
            "Point Noise",
            min=0,
            max=0.02,
            step=0.0005,
            initial_value=0,
        )

        def show_mask():
            if not checkbox_handle.value:
                mask_name = drop_down_handle.value
                flag = False 
                if mask_name == "Mask-1":
                    if mask_res[0] is not None:
                        points_handle[0].colors = mask2color(mask_res[0])
                        flag = True
                elif mask_name == "Mask-2":
                    if mask_res[1] is not None:
                        points_handle[0].colors = mask2color(mask_res[1])
                        flag = True
                elif mask_name == "Mask-3":
                    if mask_res[2] is not None:
                        points_handle[0].colors = mask2color(mask_res[2])
                        flag = True

                if iou_res[0] is not None:
                    text = "IOU: "
                    for i in range(3):
                        if best[0] == i:
                            text += f"<font color=\"red\">{i+1}: {iou_res[i]:.3f}</font> "
                        else:
                            text += f"{i+1}: {iou_res[i]:.3f} "
                    text += '\n\n'
                    text += "Org IOU: "
                    for i in range(3):
                        text += f"{i+1}: {iou_org[i]:.3f} "
                    labels = np.asarray(point_prompt_labels[0], dtype=np.int8)
                    text += "\n\n"
                    text += f"Prompts: +{np.sum(labels >= 0)} / -{np.sum(labels < 0)}"
                    markdown_handle.content = text
                else:
                    markdown_handle.content = "IOU: 1: 0.000, 2: 0.000, 3: 0.000"

                if not flag:
                    points_handle[0].colors = show_colors[0]
            else:
                points_handle[0].colors = colors_pca[0]

        def add_point_prompt(select_point, label, prompt_idx):
            prompt_color = PROMPT_COLOR if label >= 0 else NEG_PROMPT_COLOR
            server.scene.add_icosphere(
                name=f"/prompt_sphere_{prompt_idx}",
                radius=0.01,
                color=prompt_color,
                position=select_point,
            )
        

        @click_button_handle.on_click
        def _(_):
            click_button_handle.disabled = True

            @client.scene.on_pointer_event(event_type="click")
            def _(event: viser.ScenePointerEvent) -> None:
                o = np.array(event.ray_origin)
                d = np.array(event.ray_direction)
                
                A = points[0] - o 
                B = np.expand_dims(d, axis=0)
                AB = np.sum(A * B, axis=-1)
                B_squre = np.sum(B ** 2, axis=-1)
                t = AB / B_squre
                intersect_points = o + t.reshape(-1, 1) * d
                distv = np.sum((intersect_points - points[0]) ** 2, axis=-1) ** 0.5
                disth = t*np.sqrt(B_squre)
                mask = (distv < POINT_SIZE)
                if np.sum(mask) == 0:
                    mask = (distv < POINT_SIZE*5)
                    if np.sum(mask) == 0:
                        return
                select_points = points[0][mask]
                disth = disth[mask]
                min_disth_idx = np.argmin(disth)
                select_point = select_points[min_disth_idx]
                label = 1 if prompt_mode_handle.value == "Include" else -1
                print(f"选择点: {select_point}, label={label}")
                point_prompts[0].append(select_point)
                point_prompt_labels[0].append(label)
                add_point_prompt(select_point, label, len(point_prompts[0]) - 1)

                (
                    pred_mask_1,
                    pred_mask_2,
                    pred_mask_3,
                    pred_iou_1,
                    pred_iou_2,
                    pred_iou_3,
                    org_iou_1,
                    org_iou_2,
                    org_iou_3,
                    pos_masks,
                    neg_masks,
                ) = get_mask(
                    model,
                    feats[0],
                    points[0],
                    np.asarray(point_prompts[0], dtype=np.float32),
                    np.asarray(point_prompt_labels[0], dtype=np.int8),
                    return_components=True,
                )
                mask_res[0] = pred_mask_1
                mask_res[1] = pred_mask_2
                mask_res[2] = pred_mask_3
                positive_mask_res[0] = pos_masks[0]
                positive_mask_res[1] = pos_masks[1]
                positive_mask_res[2] = pos_masks[2]
                negative_mask_res[0] = neg_masks[0]
                negative_mask_res[1] = neg_masks[1]
                negative_mask_res[2] = neg_masks[2]
                iou_res[0] = pred_iou_1
                iou_res[1] = pred_iou_2
                iou_res[2] = pred_iou_3
                iou_org[0] = org_iou_1
                iou_org[1] = org_iou_2
                iou_org[2] = org_iou_3
                best[0] = np.argmax(np.array([pred_iou_1, pred_iou_2, pred_iou_3]))
                if best[0] == 0:
                    mask_name = "Mask-1"
                elif best[0] == 1:
                    mask_name = "Mask-2"
                elif best[0] == 2:
                    mask_name = "Mask-3"
                drop_down_handle.value = mask_name

                print(
                    '获取mask成功',
                    np.sum(mask_res[0]),
                    np.sum(mask_res[1]),
                    np.sum(mask_res[2]),
                    f"positive={np.sum(np.asarray(point_prompt_labels[0]) >= 0)}",
                    f"negative={np.sum(np.asarray(point_prompt_labels[0]) < 0)}",
                )
                print('获取iou成功', pred_iou_1, pred_iou_2, pred_iou_3)
                print('最佳mask', best[0]+1)

                client.scene.remove_pointer_callback()

            @client.scene.on_pointer_callback_removed
            def _():
                click_button_handle.disabled = False

        @drop_down_handle.on_update
        def _(_):
            show_mask()

        @checkbox_handle.on_update
        def _(_):
            show_mask()
        
        @ checkbox_handle_2.on_update
        def _(_):
            load_pc(use_normal=checkbox_handle_2.value, noise_std=slider_handle_2.value)
            show_mask()
        
        @clear_button_handle.on_click
        def _(_):
            clear_state()
            show_mask()

        @save_button_handle.on_click
        def _(_):
            """
            保存当前选中的 mask（与 IoU 最佳头一致）。
            先用 mask 点对原始 mesh face 做投票得到干净 seed faces，
            再用这些 seed faces 的 bbox 从原始 mesh 裁出完整密度的 faces。
            """
            if best[0] is None or mask_res[best[0]] is None:
                return
            if mesh_obj[0] is None or face_idx_arr[0] is None:
                return

            _mesh = mesh_obj[0]
            _face_idx = face_idx_arr[0]   # (N,) 采样点对应的 face 索引
            _mask = mask_res[best[0]]      # (N,) bool，每点 mask
            _positive_mask = positive_mask_res[best[0]]
            _negative_mask = negative_mask_res[best[0]]
            if _positive_mask is None:
                _positive_mask = _mask
            if _negative_mask is None:
                _negative_mask = np.zeros_like(_mask, dtype=bool)
            _prompt_points = np.asarray(point_prompts[0], dtype=np.float32).copy()
            _prompt_labels = np.asarray(point_prompt_labels[0], dtype=np.int8).copy()

            _face_ids, _positive_seed_face_ids = bbox_face_ids_from_point_mask(
                _mesh,
                _face_idx,
                _positive_mask,
                bbox_padding=args.part_bbox_padding,
            )

            if len(_positive_seed_face_ids) == 0:
                print("[Save] No faces selected, skip.")
                return

            print(f"[Save] mask 点数: {np.sum(_mask)}/{len(_mask)}")
            print(f"[Save] positive seed faces: {len(_positive_seed_face_ids)}/{len(_mesh.faces)}")
            print(f"[Save] positive bbox faces: {len(_face_ids)}/{len(_mesh.faces)}")

            if np.any(_negative_mask):
                _negative_face_ids, _negative_seed_face_ids = bbox_face_ids_from_point_mask(
                    _mesh,
                    _face_idx,
                    _negative_mask,
                    bbox_padding=args.part_bbox_padding,
                )
                if len(_negative_face_ids) > 0:
                    _before_subtract = len(_face_ids)
                    _face_ids = np.setdiff1d(_face_ids, _negative_face_ids, assume_unique=False)
                    print(
                        f"[Save] negative seed faces: {len(_negative_seed_face_ids)}/{len(_mesh.faces)}"
                    )
                    print(
                        f"[Save] negative bbox subtract: {_before_subtract} -> {len(_face_ids)}"
                    )

            if len(_face_ids) == 0:
                print("[Save] No faces selected for export, skip.")
                return

            _mesh_save, _vertex_ids = build_submesh_from_face_ids(_mesh, _face_ids)
            print(f"[Save] export 顶点数: {len(_vertex_ids)}/{len(_mesh.vertices)}")
            print(f"[Save] export 面数: {len(_face_ids)}/{len(_mesh.faces)}")

            # Step 6: 每次保存自增计数器，移除旧 UI 后重建
            save_counter[0] += 1

            # 移除上一次保存的 UI 元素
            remove_save_ui()

            name_input_handle = client.gui.add_text(
                label="file name",
                initial_value="part",
            )

            def _on_save(_):
                _user_name = name_input_handle.value.strip()
                if not _user_name:
                    return

                # Save to the directory of the currently loaded object GLB
                if cur_glb_path[0]:
                    _save_dir = os.path.dirname(cur_glb_path[0])
                else:
                    _save_dir = args.data_dir
                os.makedirs(_save_dir, exist_ok=True)
                _save_path = os.path.join(_save_dir, f"{_user_name}.glb")

                _mesh_save.export(_save_path)
                print(f"[Save] 导出部件到: {_save_path}")
                print(f"[Save] 顶点数={len(_mesh_save.vertices)}, "
                      f"面数={len(_face_ids)}/{len(_mesh.faces)}, "
                      f"iou={iou_res[best[0]]:.3f}")

                _face_ids_arr = -np.ones(len(_mesh.faces), dtype=np.int64)
                _face_ids_arr[_face_ids] = save_counter[0]
                _mask_info_path = _save_path.replace('.glb', '_mask_info.npz')
                np.savez_compressed(
                    _mask_info_path,
                    face_ids=_face_ids_arr,
                    point_mask=_mask.astype(np.uint8),
                    positive_point_mask=_positive_mask.astype(np.uint8),
                    negative_point_mask=_negative_mask.astype(np.uint8),
                    face_idx=_face_idx,
                    prompt_point=_prompt_points[0] if len(_prompt_points) else np.zeros(3, dtype=np.float32),
                    prompt_points=_prompt_points,
                    prompt_labels=_prompt_labels,
                    iou=float(iou_res[best[0]]),
                )
                print(f"[Save] Mask info saved to: {_mask_info_path}")

                # 移除保存 UI 元素
                remove_save_ui()

                clear_state()
                show_mask()

            def _on_cancel(_):
                # 移除取消的 UI 元素
                remove_save_ui()

            confirm_btn = client.gui.add_button(
                label="save", color=(0, 200, 83)
            )
            confirm_btn.on_click(_on_save)

            cancel_btn = client.gui.add_button(
                label="cancel", color=(200, 50, 50)
            )
            cancel_btn.on_click(_on_cancel)

            # 保存当前 UI 的所有 handle，以便下次点击 save 时统一移除
            save_ui_handles[0] = [name_input_handle, confirm_btn, cancel_btn]

        @slider_handle.on_update
        def _(_):
            global POINT_SIZE
            points_handle[0].point_size = slider_handle.value
            POINT_SIZE = slider_handle.value
        
        @slider_handle_2.on_update
        def _(_):
            load_pc(use_normal=checkbox_handle_2.value, noise_std=slider_handle_2.value)
            show_mask()
        
        @data_list_handle.on_update
        def _(_):
            sel = data_list_handle.value
            if not sel or sel not in _all_glbs:
                return
            load_pc(_all_glbs[sel], checkbox_handle_2.value, slider_handle_2.value)
            show_mask()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--ckpt_path', type=str, default=None, help='path to continue ckpt')
    argparser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    argparser.add_argument("--port", default=8080, type=int, help="Port to bind to")
    argparser.add_argument("--point_num", default=500000, type=int, help="Number of points to sample from the mesh (default 500k for dense coverage)")
    argparser.add_argument("--data_dir", default='../assets', type=str, help="Data directory")
    argparser.add_argument("--no_normal", action='store_true', help="Do not use normal information")
    argparser.add_argument("--part_bbox_padding", default=0.01, type=float, help="Padding ratio for bbox crop when exporting original-density faces")
    args = argparser.parse_args()

    main(args)

'''
python app.py
python app.py --ckpt_path ../weights/p3sam.ckpt --data_dir ../assets
'''
