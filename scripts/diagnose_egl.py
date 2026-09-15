from pathlib import Path
import ctypes
import json
import os
for key in ('MUJOCO_GL','PYOPENGL_PLATFORM','NVIDIA_DRIVER_CAPABILITIES','NVIDIA_VISIBLE_DEVICES','__EGL_VENDOR_LIBRARY_FILENAMES'):
    print(key, os.environ.get(key))
for p in Path('/usr/share/glvnd/egl_vendor.d').glob('*'):
    print(str(p), p.read_text())
for lib in ('libEGL.so.1', 'libEGL_nvidia.so.0', 'libnvidia-eglcore.so.535.183.01'):
    try:
        print(lib, ctypes.CDLL(lib))
    except OSError as e:
        print(lib, str(e))
