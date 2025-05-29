import torch
import numpy as np
import os
import re
import uuid
import subprocess
import folder_paths
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class SoundGenerator:
    """Sound generation node for ComfyUI using AudioCraft's MusicGen."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Ambient electronic music with synths"}),
                "duration": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "model_size": (["small", "medium", "large", "melody"], {"default": "medium"}),
                "top_k": ("INT", {"default": 250, "min": 1, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "generate_sound"
    CATEGORY = "audio"

    def __init__(self):
        self.models = {}
        self.output_dir = os.path.join(folder_paths.get_output_directory(), "audio")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _get_model(self, model_size):
        if model_size not in self.models:
            self.models[model_size] = MusicGen.get_pretrained(model_size)
        return self.models[model_size]
    
    def generate_sound(self, prompt, duration, model_size, top_k, top_p, temperature):
        model = self._get_model(model_size)
        
        # Configure model settings
        model.set_generation_params(
            use_sampling=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            duration=duration
        )
        
        # Generate audio
        audio_output = model.generate([prompt], progress=True)
        
        # Save audio file
        audio = audio_output.cpu().numpy()[0]
        sample_rate = model.sample_rate
        
        # Clean filename and ensure uniqueness
        clean_prompt = re.sub(r'[^\w\s-]', '', prompt)[:20].strip().replace(' ', '_')
        filename = f"{clean_prompt}_{str(uuid.uuid4())[:8]}.wav"
        output_path = os.path.join(self.output_dir, filename)
        
        # Write audio file
        audio_write(
            output_path.replace(".wav", ""),
            audio,
            sample_rate,
            strategy="loudness",
            format="wav"
        )
        
        print(f"Generated audio saved to {output_path}")
        return (output_path,)

class CombineVideoAudio:
    """Node for combining video with generated audio using ffmpeg."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {}),
                "audio_path": ("STRING", {}),
                "output_format": (["mp4", "mov", "webm"], {"default": "mp4"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "combine"
    CATEGORY = "audio"
    
    def __init__(self):
        self.output_dir = os.path.join(folder_paths.get_output_directory(), "video_with_audio")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def combine(self, video_path, audio_path, output_format):
        import subprocess
        
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        output_filename = f"video_with_audio_{str(uuid.uuid4())[:8]}.{output_format}"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Run ffmpeg to combine video and audio
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Combined video with audio saved to {output_path}")
        return (output_path,)

NODE_CLASS_MAPPINGS = {
    "SoundGenerator": SoundGenerator,
    "CombineVideoAudio": CombineVideoAudio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SoundGenerator": "Sound Generator",
    "CombineVideoAudio": "Combine Video & Audio"
}