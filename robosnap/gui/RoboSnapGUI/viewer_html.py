"""Three.js articulate-object viewer HTML for the RoboSnap GUI."""

from __future__ import annotations

import json
from pathlib import Path


def _get_articulate_viewer_html(base_dir: Path, object_name: str, meshes: list):

    mesh_data = []

    for mesh in meshes:
        if not mesh["filename"].lower().endswith(".glb"):
            continue
        mesh_path = base_dir / object_name / mesh["filename"]
  
        if not mesh_path.exists():
            mesh_path = base_dir / mesh["filename"]

        if not mesh_path.exists():
            mesh_path = Path(mesh["path"])

        mesh_data.append({
            "name": mesh["name"],
            "url": "/gradio_api/file=" + str(mesh_path.absolute())
        })

    mesh_json = json.dumps(mesh_data)

    return f"""
<iframe style="width:100%;height:600px;border:none;"
srcdoc='
<html>
<body style="margin:0;background:#111">

<div id="viewer" style="width:100vw;height:100vh"></div>

<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>

<script>

console.log("three viewer start")

const meshData = {mesh_json}

const container=document.getElementById("viewer")

const scene=new THREE.Scene()
scene.background=new THREE.Color(0x222222)

const camera=new THREE.PerspectiveCamera(60,window.innerWidth/window.innerHeight,0.1,1000)
camera.position.set(2,2,2)

const renderer=new THREE.WebGLRenderer({{antialias:true}})
renderer.setSize(window.innerWidth,window.innerHeight)

renderer.outputEncoding = THREE.sRGBEncoding
renderer.toneMapping = THREE.ACESFilmicToneMapping
renderer.toneMappingExposure = 1

renderer.physicallyCorrectLights = true


container.appendChild(renderer.domElement)

const controls=new THREE.OrbitControls(camera,renderer.domElement)

scene.add(new THREE.GridHelper(10,10))
scene.add(new THREE.AxesHelper(1))

renderer.physicallyCorrectLights = true

scene.add(new THREE.AmbientLight(0xffffff,0.35))

const keyLight = new THREE.DirectionalLight(0xffffff,1.2)
keyLight.position.set(5,8,5)
scene.add(keyLight)

const fillLight = new THREE.DirectionalLight(0xffffff,0.5)
fillLight.position.set(-5,3,-5)
scene.add(fillLight)

const rimLight = new THREE.DirectionalLight(0xffffff,0.6)
rimLight.position.set(0,6,-6)
scene.add(rimLight)


const loader=new THREE.GLTFLoader()

meshData.forEach(m=>{{
 loader.load(m.url,(gltf)=>{{

gltf.scene.traverse(o=>{{
 if(o.isMesh){{

   if(o.material){{
      o.material.metalness = 0
      o.material.roughness = 0.85
   }}

 }}

}})

scene.add(gltf.scene)

}})
}})

function animate(){{
 requestAnimationFrame(animate)
 controls.update()
 renderer.render(scene,camera)
}}

animate()

</script>

</body>
</html>
'>
</iframe>
"""  

