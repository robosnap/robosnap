"""
Articulate Object Manager
Manages mesh objects and joint annotations for IsaacSim USD format.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any


class ArticulateObjectManager:
    """
    Manages articulate objects: scans segmented objects, loads meshes,
    and saves joint definitions in USD-compatible format.
    """
    
    def __init__(self, base_dir: Path):
        """
        Args:
            base_dir: Base directory containing the segmented objects
        """
        self.base_dir = Path(base_dir)
        self.objects_dir = self.base_dir
        self.joints_dir = self.base_dir  # joints stored alongside objects
        
    def scan_segmented_objects(self, base_path: str = None) -> List[Dict[str, Any]]:
        """
        Scan the base directory for segmented objects.
        
        Args:
            base_path: Optional custom path to scan. If None, uses self.objects_dir.
        
        Returns:
            List of object info dicts with keys: name, path, has_joints, mesh_files
        """
        objects = []
        
        # Use custom path if provided, otherwise use default
        scan_dir = Path(base_path) if base_path else self.objects_dir
        
        if not scan_dir.exists():
            return objects
            
        for item in scan_dir.iterdir():
            if not item.is_dir():
                continue
                
            # Skip special directories (including 'background')
            if item.name.startswith('.') or item.name.startswith('_'):
                continue
            if item.name.lower() == 'background':
                continue
                
            # Check if it's an object directory 
            # (has .png/.glb files directly OR has all_mask/top*_mask subdirs OR has background_only.mp4)
            png_files = list(item.glob("*.png"))
            glb_files = list(item.glob("*.glb"))
            
            # Also check subdirectories for masks
            has_all_mask = (item / "all_mask").exists()
            top_mask_dirs = [p for p in item.glob("top*_mask") if p.is_dir()]
            has_top_mask = bool(top_mask_dirs)
            has_background_video = (item / "background_only.mp4").exists()
            
            # Object exists if: has direct png/glb, OR has mask subdirs, OR has background video
            if png_files or glb_files or has_all_mask or has_top_mask or has_background_video:
                # Check if joints exist for this object
                joints_file = item / f"{item.name}_joints.json"
                has_joints = joints_file.exists()
                
                # Count total mask files
                mask_count = len(png_files) + len(glb_files)
                if has_all_mask:
                    mask_count += len(list((item / "all_mask").glob("*.png")))
                for top_mask_dir in top_mask_dirs:
                    mask_count += len(list(top_mask_dir.glob("*.png")))
                
                obj_info = {
                    "name": item.name,
                    "path": str(item),
                    "has_joints": has_joints,
                    "mesh_count": mask_count,
                    "png_files": [f.name for f in png_files],
                    "glb_files": [f.name for f in glb_files],
                }
                objects.append(obj_info)
                
        return sorted(objects, key=lambda x: x["name"])
    
    def scan_single_mask_objects(self, base_path: str = None) -> List[Dict[str, Any]]:
        """
        Scan a directory with GLB files directly in the folder (like single_mask/).
        This is for the case where meshes are named 0.glb, 1.glb, 2.glb, etc.
        
        Args:
            base_path: Path to scan (e.g., case1/single_mask)
            
        Returns:
            List of object info dicts
        """
        objects = []
        scan_dir = Path(base_path) if base_path else self.objects_dir
        
        if not scan_dir.exists():
            return objects
        
        # Look for GLB files directly in the folder (not in subdirectories)
        for glb_file in sorted(scan_dir.glob("*.glb")):
            # Skip scene_composed.glb
            if glb_file.name == "scene_composed.glb":
                continue
            
            obj_name = glb_file.stem  # "0", "1", "2", etc.
            
            # Check if joints exist
            joints_file = scan_dir / f"{obj_name}_joints.json"
            has_joints = joints_file.exists()
            
            # Check for PLY file too
            ply_file = scan_dir / f"{obj_name}.ply"
            
            obj_info = {
                "name": obj_name,
                "path": str(scan_dir),
                "has_joints": has_joints,
                "mesh_count": 1 + (1 if ply_file.exists() else 0),
                "glb_file": glb_file.name,
                "ply_file": ply_file.name if ply_file.exists() else None,
            }
            objects.append(obj_info)
        
        return sorted(objects, key=lambda x: x["name"])
    
    def get_object_meshes(self, object_name: str, base_path: str = None) -> List[Dict[str, Any]]:
        """
        Get mesh files for a specific object.
        
        Args:
            object_name: Name of the object directory
            base_path: Optional custom base path. If None, uses self.objects_dir.
            
        Returns:
            List of mesh file info
        """
        # Use custom base path if provided
        base_dir = Path(base_path) if base_path else self.objects_dir
        obj_dir = base_dir / object_name
        if not obj_dir.exists():
            return []
            
        meshes = []
        
        # Look for GLB files first (preferred)
        for glb_file in sorted(obj_dir.glob("*.glb")):
            meshes.append({
                "name": glb_file.stem,
                "filename": glb_file.name,
                "path": str(glb_file),
                "type": "glb"
            })
            
        # Also look for PLY files
        for ply_file in sorted(obj_dir.glob("*.ply")):
            meshes.append({
                "name": ply_file.stem,
                "filename": ply_file.name,
                "path": str(ply_file),
                "type": "ply"
            })
            
        return meshes
    
    def save_joints(
        self, 
        object_name: str, 
        joints: List[Dict[str, Any]],
        mesh_info: Dict[str, Dict[str, Any]] = None,
        base_path: str = None
    ) -> bool:
        """
        Save joint definitions to JSON file in USD-compatible format.
        
        Args:
            object_name: Name of the object
            joints: List of joint definitions
            mesh_info: Optional dict mapping mesh names to their world transforms
            base_path: Optional custom base path. If None, uses self.objects_dir.
            
        Returns:
            True if successful
        """
        base_dir = Path(base_path) if base_path else self.objects_dir
        obj_dir = base_dir / object_name
        obj_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert joints to USD-compatible format
        usd_joints = self._convert_to_usd_format(joints, mesh_info or {})
        
        # Save joints file
        joints_file = obj_dir / f"{object_name}_joints.json"
        with open(joints_file, 'w') as f:
            json.dump(usd_joints, f, indent=2)
            
        # Also create a .joints.usd text format (for direct IsaacSim loading)
        usd_text_file = obj_dir / f"{object_name}_joints.usd"
        self._save_usd_text_format(usd_text_file, usd_joints)
        
        return True
    
    def _convert_to_usd_format(
        self, 
        joints: List[Dict[str, Any]],
        mesh_info: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Convert joints to USD-compatible format.
        
        USD Joint Format:
        - RevoluteJoint: rotation around an axis
        - PrismaticJoint: translation along an axis
        - FixedJoint: rigid connection
        """
        usd_joints = {
            "version": "1.0",
            "object_type": "articulated_object",
            "joints": []
        }
        
        for i, joint in enumerate(joints):
            joint_type = joint.get("type", "FixedJoint")
            
            # Parse position
            pos = joint.get("position", {})
            if isinstance(pos, dict):
                pos = [pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)]
            else:
                pos = [0, 0, 0]
            
            # Convert limits to radians if revolute
            lower = joint.get("lowerLimit", -90)
            upper = joint.get("upperLimit", 90)
            axis = joint.get("axis", "Y")
            
            if joint_type == "RevoluteJoint":
                # Convert degrees to radians for USD
                lower_rad = np.deg2rad(lower)
                upper_rad = np.deg2rad(upper)
            else:
                # For prismatic, keep as is (assumed to be in meters)
                lower_rad = lower
                upper_rad = upper
            
            usd_joint = {
                "id": joint.get("id", i),
                "type": joint_type,
                "body0": joint.get("body0", ""),
                "body1": joint.get("body1", ""),
                "localPos0": {
                    "x": pos[0],
                    "y": pos[1], 
                    "z": pos[2]
                },
                "localPos1": {
                    "x": pos[0],
                    "y": pos[1],
                    "z": pos[2]
                },
                "axis": axis,
                "lowerLimit": lower_rad,
                "upperLimit": upper_rad,
                "friction": joint.get("friction", 0.2),
                "breakForce": joint.get("breakForce", "inf"),
                "breakTorque": joint.get("breakTorque", "inf")
            }
            
            usd_joints["joints"].append(usd_joint)
            
        return usd_joints
    
    def _save_usd_text_format(self, output_file: Path, joints_data: Dict[str, Any]):
        """
        Save joints in USD text format for direct IsaacSim loading.
        
        This creates a .usd file that can be directly loaded by IsaacSim.
        """
        lines = []
        lines.append('#usda 1.0')
        lines.append('(')
        lines.append('    defaultPrim = "World"')
        lines.append(')')
        lines.append('')
        lines.append('def World (')
        lines.append('    kind = "component"')
        lines.append(')')
        lines.append('{')
        
        for joint in joints_data.get("joints", []):
            joint_id = joint.get("id", 0)
            joint_type = joint.get("type", "FixedJoint")
            
            # USD joint type
            usd_type = "FixedJoint"
            if joint_type == "RevoluteJoint":
                usd_type = "RevoluteJoint"
            elif joint_type == "PrismaticJoint":
                usd_type = "PrismaticJoint"
                
            lines.append(f'    def "{joint_type}_{joint_id}" ({usd_type})')
            lines.append('    {')
            
            # Body paths
            body0 = joint.get("body0", "")
            body1 = joint.get("body1", "")
            lines.append(f'        rel body0 = @{body0}@')
            lines.append(f'        rel body1 = @{body1}@')
            
            # Position
            pos0 = joint.get("localPos0", {})
            lines.append(f'        double3 localPos0 = ({pos0.get("x", 0)}, {pos0.get("y", 0)}, {pos0.get("z", 0)})')
            
            pos1 = joint.get("localPos1", {})
            lines.append(f'        double3 localPos1 = ({pos1.get("x", 0)}, {pos1.get("y", 0)}, {pos1.get("z", 0)})')
            
            # Limits
            lower = joint.get("lowerLimit", -1.57)
            upper = joint.get("upperLimit", 1.57)
            lines.append(f'        float limit:lower = {lower}')
            lines.append(f'        float limit:upper = {upper}')
            
            # Axis
            axis = joint.get("axis", "Y")
            lines.append(f'        token axis = "{axis}"')
            
            lines.append('    }')
            lines.append('')
            
        lines.append('}')
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
    
    def load_joints(self, object_name: str, base_path: str = None) -> Optional[Dict[str, Any]]:
        """
        Load joint definitions for an object.
        
        Args:
            object_name: Name of the object
            base_path: Optional custom base path. If None, uses self.objects_dir.
            
        Returns:
            Joint data dict or None if not found
        """
        base_dir = Path(base_path) if base_path else self.objects_dir
        joints_file = base_dir / object_name / f"{object_name}_joints.json"
        if not joints_file.exists():
            return None
            
        with open(joints_file, 'r') as f:
            return json.load(f)
    
    def delete_joints(self, object_name: str) -> bool:
        """
        Delete joint definitions for an object.
        
        Args:
            object_name: Name of the object
            
        Returns:
            True if successful
        """
        joints_file = self.objects_dir / object_name / f"{object_name}_joints.json"
        usd_file = self.objects_dir / object_name / f"{object_name}_joints.usd"
        
        deleted = False
        if joints_file.exists():
            joints_file.unlink()
            deleted = True
        if usd_file.exists():
            usd_file.unlink()
            
        return deleted
    
    def get_mesh_url(self, object_name: str, filename: str, server_url: str = None) -> str:
        """
        Get HTTP URL for a mesh file.
        
        Args:
            object_name: Name of the object directory
            filename: Name of the mesh file
            server_url: Optional server URL prefix
            
        Returns:
            URL string for the mesh file
        """
        if server_url:
            return f"{server_url}/{object_name}/{filename}"
        
        # Use file:// protocol for local files
        mesh_path = self.objects_dir / object_name / filename
        return str(mesh_path.absolute().as_uri())
    
    def create_articulate_definition(
        self, 
        object_name: str, 
        joints: List[Dict[str, Any]],
        mesh_files: List[str],
        output_dir: Path = None
    ) -> Dict[str, Any]:
        """
        Create a complete articulate object definition.
        
        This creates:
        1. joints.json - joint definitions
        2. object_info.json - object metadata
        3. (Optional) .usd file for IsaacSim
        
        Args:
            object_name: Name of the object
            joints: List of joint definitions
            mesh_files: List of mesh filenames
            output_dir: Output directory (defaults to object_dir)
            
        Returns:
            Definition info dict
        """
        output_dir = output_dir or (self.objects_dir / object_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save joints
        self.save_joints(object_name, joints)
        
        # Create object info
        object_info = {
            "name": object_name,
            "mesh_files": mesh_files,
            "joints_file": f"{object_name}_joints.json",
            "usd_file": f"{object_name}_joints.usd",
            "created": True
        }
        
        info_file = output_dir / "object_info.json"
        with open(info_file, 'w') as f:
            json.dump(object_info, f, indent=2)
            
        return object_info


def create_usd_joint_definition(
    joint_type: str,
    body0_path: str,
    body1_path: str,
    position: List[float],
    axis: str = "Y",
    lower_limit: float = -1.57,
    upper_limit: float = 1.57,
    friction: float = 0.2
) -> Dict[str, Any]:
    """
    Create a single USD joint definition.
    
    Args:
        joint_type: "RevoluteJoint", "PrismaticJoint", or "FixedJoint"
        body0_path: Path to first body (parent)
        body1_path: Path to second body (articulated part)
        position: [x, y, z] local position
        axis: Rotation/translation axis ("X", "Y", or "Z")
        lower_limit: Lower limit (radians for revolute, meters for prismatic)
        upper_limit: Upper limit (radians for revolute, meters for prismatic)
        friction: Joint friction
        
    Returns:
        Joint definition dict
    """
    return {
        "type": joint_type,
        "body0": body0_path,
        "body1": body1_path,
        "localPos0": {"x": position[0], "y": position[1], "z": position[2]},
        "localPos1": {"x": position[0], "y": position[1], "z": position[2]},
        "axis": axis,
        "lowerLimit": lower_limit,
        "upperLimit": upper_limit,
        "friction": friction,
        "breakForce": "inf",
        "breakTorque": "inf"
    }
