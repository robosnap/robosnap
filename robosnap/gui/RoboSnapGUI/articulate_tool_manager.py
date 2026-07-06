"""Articulate tool subprocess management for the RoboSnap GUI."""

import os
import socket
import subprocess
import threading
import time
from pathlib import Path


def _clear_socks_proxy_env(env=None):
    env = os.environ if env is None else env
    if env.get("ROBOSNAP_KEEP_PROXY") == "1":
        return env
    for proxy_var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        value = env.get(proxy_var, "")
        if value.lower().startswith("socks"):
            env.pop(proxy_var, None)
    return env


def _derive_conda_prefix(python_executable, env_name=None):
    if env_name:
        env_names = (env_name,) if isinstance(env_name, str) else tuple(env_name)
        for name in env_names:
            value = os.environ.get(name)
            if value:
                return value
    if not python_executable:
        return None
    path = Path(python_executable).expanduser()
    if not path.is_absolute():
        return None
    parts = path.parts
    if len(parts) >= 2 and parts[-2] == "bin" and parts[-1].startswith("python"):
        return str(path.parents[1])
    return None


def _subprocess_env_for_python(python_executable, *, conda_env_name=None, pythonpath=None):
    env = _clear_socks_proxy_env(os.environ.copy())
    prefix = _derive_conda_prefix(python_executable, conda_env_name)
    if prefix:
        env["CONDA_PREFIX"] = prefix
    if pythonpath:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{existing}" if existing else str(pythonpath)
    return env


def find_free_port(start_port=8080, max_attempts=100):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError("Could not find a free port")


class ArticulateToolManager:
    """
    Manages articulate app instances for each object.
    Launches subprocess for each object and provides iframe URL.
    """
    def __init__(self, python="python", app=None, ckpt=None, base_port=8180):
        self.processes = {}  # object_name -> subprocess
        self.ports = {}      # object_name -> port
        self.python = python
        self.app = app
        self.ckpt = ckpt
        self.base_port = int(base_port)
        self.starting = {}  # object_name -> bool (True if still starting)
        self.start_lock = threading.Lock()

    def configure(self, python=None, app=None, ckpt=None, base_port=None):
        self.python = python or self.python
        self.app = app or self.app
        self.ckpt = ckpt or self.ckpt
        if base_port is not None:
            self.base_port = int(base_port)

    def missing_requirements(self):
        missing = []
        if not self.app or not Path(self.app).expanduser().exists():
            missing.append(f"Articulate tool app not found: {self.app}")
        if not self.ckpt or not Path(self.ckpt).expanduser().exists():
            missing.append(f"Articulate tool checkpoint not found: {self.ckpt}")
        return missing

    def _get_object_data_dir(self, base_dir, object_name):
        """Get the data directory path for an object"""
        # basedir/multi_mask/object_name
        return str(Path(base_dir) / object_name)

    def _start_server_background(self, object_name, base_dir, port):
        """Background thread to start the server"""
        data_dir = self._get_object_data_dir(base_dir, object_name)
        cmd = [
            self.python,
            self.app,
            "--ckpt_path", self.ckpt,
            "--data_dir", data_dir,
            "--host", "0.0.0.0",
            "--port", str(port)
        ]

        print(f"[ArticulateTool] Starting server thread for {object_name} on port {port}")
        # Set cwd to demo directory so that sys.path.append('..') works correctly
        app_dir = os.path.dirname(cmd[1])
        env = _subprocess_env_for_python(self.python, conda_env_name=("PY_ARTICULATE_CONDA_PREFIX", "PY_P3SAM_CONDA_PREFIX"))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=app_dir)
        self.processes[object_name] = proc

        # Wait for server to be ready
        max_wait = 60
        waited = 0
        while waited < max_wait:
            time.sleep(2)
            waited += 2

            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                print(f"[ArticulateTool] Process died: {stderr.decode()[-500:]}")
                with self.start_lock:
                    self.starting[object_name] = False
                return

            try:
                import socket as sock
                test_sock = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
                test_sock.settimeout(1)
                result = test_sock.connect_ex(('127.0.0.1', port))
                test_sock.close()
                if result == 0:
                    print(f"[ArticulateTool] Server ready for {object_name} on port {port} after {waited}s")
                    with self.start_lock:
                        self.starting[object_name] = False
                    return
            except:
                pass

        print(f"[ArticulateTool] Server startup timeout for {object_name} after {max_wait}s")
        with self.start_lock:
            self.starting[object_name] = False

    def launch_for_object(self, object_name, base_dir):
        """Launch the articulate app for a specific object (non-blocking)"""
        # Stop existing process for this object if any
        self.stop_object(object_name)

        # Find free port
        port = find_free_port(self.base_port)
        self.ports[object_name] = port

        # Mark as starting
        with self.start_lock:
            self.starting[object_name] = True

        # Start server in background thread
        thread = threading.Thread(target=self._start_server_background, args=(object_name, base_dir, port))
        thread.daemon = True
        thread.start()

        print(f"[ArticulateTool] Launched background thread for {object_name}, port {port}")
        return port

    def get_iframe_url(self, object_name):
        """Get the iframe URL for a running articulate instance"""
        port = self.ports.get(object_name)
        if port is None:
            return None
        return self.get_public_url(port)

    def get_public_url(self, port):
        """
        URL that the browser should use to reach the articulate service.

        The articulate process itself listens inside this remote container, but the
        browser may be running on a local laptop or through a DSW proxy. In that
        case, 127.0.0.1 in the iframe means the browser machine, not this
        container. Set the public URL template environment variable when the
        external URL differs from the default local-forwarding URL.
        """
        template = os.environ.get("ARTICULATE_PUBLIC_URL_TEMPLATE") or os.environ.get("P3SAM_PUBLIC_URL_TEMPLATE")
        if template:
            return template.format(port=port)
        return f"http://127.0.0.1:{port}"

    def stop_object(self, object_name):
        """Stop the articulate process for an object"""
        if object_name in self.processes:
            proc = self.processes[object_name]
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            del self.processes[object_name]
        if object_name in self.ports:
            del self.ports[object_name]

    def stop_all(self):
        """Stop all articulate processes"""
        for obj_name in list(self.processes.keys()):
            self.stop_object(obj_name)

