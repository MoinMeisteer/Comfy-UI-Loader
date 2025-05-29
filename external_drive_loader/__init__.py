import os
import platform
import folder_paths
import comfy
import shutil
import importlib

from .checkpoint_loader import ExternalCheckpointLoader
from .lora_loader import ExternalLoRALoader
from .utils import add_temp_path

def check_compatibility():
    try:
        import comfy
        if hasattr(comfy, 'version'):
            print(f"ComfyUI Version: {comfy.version}")
        return True
    except:
        print("Warnung: Konnte ComfyUI Version nicht ermitteln")
        return True

# ComfyUI-Kompatibilität prüfen
is_compatible = check_compatibility()

NODE_CLASS_MAPPINGS = {
    "ExternalCheckpointLoader": ExternalCheckpointLoader,
    "ExternalLoRALoader": ExternalLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExternalCheckpointLoader": "External Checkpoint Loader",
    "ExternalLoRALoader": "External LoRA Loader",
}