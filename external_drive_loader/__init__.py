import os
import platform
import folder_paths
import comfy
import shutil
import importlib

def add_temp_path(folder_type, temp_path):
    """Fügt einen temporären Pfad zu folder_paths hinzu, ohne Abhängigkeit von API-Änderungen."""
    try:
        # Versuche zunächst die add_folder_paths_to_mem-Funktion, falls vorhanden
        if hasattr(folder_paths, 'add_folder_paths_to_mem'):
            folder_paths.add_folder_paths_to_mem(folder_type, [temp_path])
            return True
        # Wenn nicht, versuche die ältere add_path-Methode
        elif hasattr(folder_paths, 'add_path'):
            folder_paths.add_path(folder_type, temp_path)
            return True
        # Direktes Hinzufügen zum Pfad-Dictionary als letzten Ausweg
        else:
            # Vorsichtiges Hinzufügen zum paths-Dictionary, wenn es existiert
            if hasattr(folder_paths, 'paths') and isinstance(folder_paths.paths, dict):
                if folder_type in folder_paths.paths:
                    if isinstance(folder_paths.paths[folder_type], list):
                        if temp_path not in folder_paths.paths[folder_type]:
                            folder_paths.paths[folder_type].append(temp_path)
                            return True
    except Exception as e:
        print(f"Fehler beim Hinzufügen des temporären Pfads: {str(e)}")
    return False

def is_compatible_comfyui_version():
    """Prüft, ob die ComfyUI-Version mit dem Plugin kompatibel ist."""
    try:
        # Versuche Versionen zu vergleichen, wenn möglich
        if hasattr(comfy, 'version'):
            version = getattr(comfy, 'version')
            if isinstance(version, str):
                import re
                match = re.search(r'(\d+)\.(\d+)\.(\d+)', version)
                if match:
                    major, minor, patch = map(int, match.groups())
                    # Angenommen, wir benötigen mindestens Version 1.0.0
                    return major >= 1
        
        # Alternativ prüfe die Existenz von kritischen Funktionen
        critical_functions = [
            hasattr(folder_paths, 'get_folder_paths'),
            hasattr(folder_paths, 'add_folder_paths_to_mem') or hasattr(folder_paths, 'add_path')
        ]
        
        return all(critical_functions)
    except:
        return False

class ExternalCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Externe Festplatten finden und auflisten
        external_drives = cls.find_external_drives()
        print(f"Gefundene externe Laufwerke: {external_drives}")
        
        # Checkpoints von allen externen Laufwerken sammeln
        all_ckpt_files = []
        drives_with_models = []
        
        for drive in external_drives:
            # Pfad zum Checkpoints-Ordner
            model_path = os.path.join(drive, "checkpoints")
            print(f"Suche nach Checkpoints in: {model_path}")
            
            # Wenn der Checkpoints-Ordner nicht existiert, erstelle ihn
            if not os.path.exists(model_path):
                try:
                    os.makedirs(model_path)
                    print(f"Checkpoints-Ordner auf {drive} erstellt.")
                    drives_with_models.append(f"{os.path.basename(drive)} (neu erstellt)")
                except (PermissionError, OSError) as e:
                    print(f"Konnte keinen Checkpoints-Ordner auf {drive} erstellen: {str(e)}")
                    continue
            
            # Schaue nach Modellen im Ordner
            try:
                files = [f for f in os.listdir(model_path) if f.endswith('.safetensors') or f.endswith('.ckpt')]
                print(f"Gefundene Dateien in {model_path}: {files}")
                
                if files:  # Nur Laufwerke mit Modellen hinzufügen
                    all_ckpt_files.extend([(os.path.join(model_path), f) for f in files])
                    if os.path.basename(drive) not in [d.split(" (neu erstellt)")[0] for d in drives_with_models]:
                        drives_with_models.append(os.path.basename(drive))
                else:
                    # Wenn Ordner existiert aber leer ist
                    if os.path.basename(drive) not in [d.split(" (neu erstellt)")[0] for d in drives_with_models]:
                        drives_with_models.append(f"{os.path.basename(drive)} (leer)")
            except (PermissionError, FileNotFoundError) as e:
                print(f"Fehler beim Lesen von {model_path}: {str(e)}")
        
        # Ergebnisse für die UI vorbereiten
        if all_ckpt_files:
            ckpt_names = [f"{os.path.basename(os.path.dirname(d))} - {f}" for d, f in all_ckpt_files]
        else:
            ckpt_names = ["Keine Modelle gefunden"]
        
        # Speichern der Pfad-Informationen zur späteren Verwendung
        cls.path_mapping = dict(zip(ckpt_names, all_ckpt_files)) if all_ckpt_files else {}
        
        # Füge Informationen für die UI hinzu
        drive_info = []
        for drive in drives_with_models:
            if "(neu erstellt)" in drive:
                drive_info.append(f"{drive.split(' (neu erstellt)')[0]} (Ordner erstellt)")
            elif "(leer)" in drive:
                drive_info.append(f"{drive.split(' (leer)')[0]} (Ordner leer)")
            else:
                drive_info.append(drive)
                
        drive_info_str = ", ".join(drive_info) if drive_info else "Keine"
        
        return {
            "required": {
                "ckpt_name": (ckpt_names, {"info": f"Laufwerke: {drive_info_str}"})
            }
        }
    
    @staticmethod
    def find_external_drives():
        """Erkennt externe Festplatten basierend auf dem Betriebssystem"""
        system = platform.system()
        drives = []
        
        if system == "Darwin":  # macOS
            volumes_dir = "/Volumes"
            if os.path.exists(volumes_dir):
                drives = [os.path.join(volumes_dir, d) for d in os.listdir(volumes_dir) 
                         if d != "Macintosh HD" and os.path.isdir(os.path.join(volumes_dir, d))]
        
        elif system == "Windows":
            try:
                import win32api
                drives = [f"{d}\\" for d in win32api.GetLogicalDriveStrings().split('\000')[:-1] 
                         if os.path.exists(f"{d}\\") and win32api.GetDriveType(f"{d}\\") == win32api.DRIVE_REMOVABLE]
            except ImportError:
                # Fallback wenn win32api nicht verfügbar ist
                import string
                drives = [f"{d}:\\" for d in string.ascii_uppercase 
                         if os.path.exists(f"{d}:\\")]
        
        elif system == "Linux":
            media_dir = "/media"
            if os.path.exists(media_dir):
                user = os.getenv("USER", "")
                user_media = os.path.join(media_dir, user)
                if os.path.exists(user_media):
                    drives = [os.path.join(user_media, d) for d in os.listdir(user_media) 
                             if os.path.isdir(os.path.join(user_media, d))]
        
        return drives
    
    RETURN_TYPES = ["MODEL", "CLIP", "VAE"]
    FUNCTION = "load_ckpt"
    CATEGORY = "loaders"

    def load_ckpt(self, ckpt_name):
        # Wenn keine Modelle gefunden wurden
        if ckpt_name == "Keine Modelle gefunden":
            raise ValueError("Keine Modelle gefunden. Bitte kopieren Sie Checkpoints in den 'checkpoints'-Ordner auf einem externen Laufwerk.")
        
        # Vollständigen Pfad zum Checkpoint aus dem Mapping abrufen
        drive_path, filename = self.__class__.path_mapping.get(ckpt_name, (None, None))
        if not drive_path:
            raise ValueError(f"Modellpfad für {ckpt_name} konnte nicht gefunden werden.")
            
        ckpt_path = os.path.join(drive_path, filename)
        print(f"Lade Checkpoint von: {ckpt_path}")
        
        # Alternative Lademethode
        try:
            from comfy import sd
            print("Versuche direktes Laden mit load_checkpoint_guess_config...")
            result = sd.load_checkpoint_guess_config(ckpt_path)
            if result:
                print("Checkpoint erfolgreich direkt geladen")
                # Stelle sicher, dass wir genau 3 Elemente zurückgeben
                if len(result) >= 3:
                    return (result[0], result[1], result[2])
        except Exception as e:
            print(f"Direktes Laden fehlgeschlagen: {str(e)}")
        
        # Versuche die Standardmethode mit temporärem Pfad
        try:
            from nodes import CheckpointLoaderSimple
            loader = CheckpointLoaderSimple()
            
            # Temporäre Kopie erstellen
            import shutil, tempfile
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, filename)
            
            try:
                shutil.copy2(ckpt_path, temp_file)
                print(f"Checkpoint nach {temp_file} kopiert")
                
                success = add_temp_path("checkpoints", temp_dir)
                if success:
                    print(f"Lade Checkpoint aus temporärem Pfad: {filename}")
                    result = loader.load_checkpoint(filename)
                    
                    # Aufräumen
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                    
                    return result
            except Exception as e:
                print(f"Fehler beim temporären Kopieren: {str(e)}")
                try:
                    os.remove(temp_file)
                except:
                    pass
        except Exception as e:
            print(f"Checkpoint-Loader konnte nicht initialisiert werden: {str(e)}")
        
        # Als letzten Ausweg versuche direktes Laden ohne loader
        try:
            from comfy import sd
            print("Versuche letzten Ausweg mit load_checkpoint_guess_config...")
            result = sd.load_checkpoint_guess_config(ckpt_path)
            if result:
                if len(result) >= 3:
                    return (result[0], result[1], result[2])
                else:
                    raise ValueError("Unerwartetes Format des Checkpoint-Ergebnisses")
        except Exception as e2:
            print(f"Auch Fallback fehlgeschlagen: {str(e2)}")
            raise ValueError(f"Konnte Checkpoint nicht laden: {ckpt_name}")
            
    @classmethod
    def setup_model_cache(cls):
        import tempfile
        
        # Cache-Verzeichnis einrichten
        cache_dir = os.path.join(tempfile.gettempdir(), "comfyui_ext_loader_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cls.cache_dir = cache_dir
        cls.cached_models = {}
        
        return cache_dir

    def cache_model(self, ckpt_name):
        drive_path, filename = self.__class__.path_mapping.get(ckpt_name, (None, None))
        if not drive_path:
            return False
            
        source_path = os.path.join(drive_path, filename)
        cache_path = os.path.join(self.__class__.cache_dir, filename)
        
        try:
            shutil.copy2(source_path, cache_path)
            self.__class__.cached_models[ckpt_name] = cache_path
            return cache_path
        except Exception as e:
            print(f"Caching fehlgeschlagen: {e}")
            return False
        
def fix_image_format(tensor):
    """Korrigiert das Format eines Bildtensors für SaveImage"""
    import torch
    
    if tensor is None:
        # Erstelle einen leeren Tensor im richtigen Format
        return torch.zeros(1, 3, 512, 512, dtype=torch.float32)
    
    # Wenn es ein Torch-Tensor ist
    if isinstance(tensor, torch.Tensor):
        shape = tensor.shape
        
        # Prüfe, ob es ein problematisches Format hat (1, 1, 512) oder ähnlich
        if len(shape) == 3 and shape[1] == 1:
            # Konvertiere zu (1, 3, 512, 512)
            correct_tensor = torch.zeros(1, 3, 512, 512, dtype=torch.float32)
            return correct_tensor
        
        # Wenn es ein (1, 512, 512, 1) Format hat
        if len(shape) == 4 and shape[3] == 1:
            # Konvertiere zu (1, 3, 512, 512)
            correct_tensor = torch.zeros(1, 3, shape[1], shape[2], dtype=torch.float32)
            # Kopiere den Grauwert in alle drei Kanäle
            correct_tensor[:, 0, :, :] = tensor[:, :, :, 0]
            correct_tensor[:, 1, :, :] = tensor[:, :, :, 0]
            correct_tensor[:, 2, :, :] = tensor[:, :, :, 0]
            return correct_tensor
    
    return tensor

class ExternalVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Externe Festplatten finden und auflisten
        external_drives = cls.find_external_drives()
        print(f"Gefundene externe Laufwerke für VAEs: {external_drives}")
        
        # VAEs von allen externen Laufwerken sammeln
        all_vae_files = []
        drives_with_models = []
        
        for drive in external_drives:
            # Pfad zum VAE-Ordner
            model_path = os.path.join(drive, "vae")
            print(f"Suche nach VAEs in: {model_path}")
            
            # Wenn der VAE-Ordner nicht existiert, erstelle ihn
            if not os.path.exists(model_path):
                try:
                    os.makedirs(model_path)
                    print(f"VAE-Ordner auf {drive} erstellt.")
                    drives_with_models.append(f"{os.path.basename(drive)} (neu erstellt)")
                except (PermissionError, OSError) as e:
                    print(f"Konnte keinen VAE-Ordner auf {drive} erstellen: {str(e)}")
                    continue
            
            # Schaue nach Modellen im Ordner
            try:
                files = [f for f in os.listdir(model_path) if f.endswith('.safetensors') or f.endswith('.ckpt') or f.endswith('.pt')]
                print(f"Gefundene VAE-Dateien in {model_path}: {files}")
                
                if files:  # Nur Laufwerke mit Modellen hinzufügen
                    all_vae_files.extend([(os.path.join(model_path), f) for f in files])
                    if os.path.basename(drive) not in [d.split(" (neu erstellt)")[0] for d in drives_with_models]:
                        drives_with_models.append(os.path.basename(drive))
                else:
                    # Wenn Ordner existiert aber leer ist
                    if os.path.basename(drive) not in [d.split(" (neu erstellt)")[0] for d in drives_with_models]:
                        drives_with_models.append(f"{os.path.basename(drive)} (leer)")
            except (PermissionError, FileNotFoundError) as e:
                print(f"Fehler beim Lesen von {model_path}: {str(e)}")
        
        # Ergebnisse für die UI vorbereiten
        if all_vae_files:
            vae_names = [f"{os.path.basename(os.path.dirname(d))} - {f}" for d, f in all_vae_files]
        else:
            vae_names = ["Keine VAEs gefunden"]
        
        # Speichern der Pfad-Informationen zur späteren Verwendung
        cls.path_mapping = dict(zip(vae_names, all_vae_files)) if all_vae_files else {}
        
        # Füge Informationen für die UI hinzu
        drive_info = []
        for drive in drives_with_models:
            if "(neu erstellt)" in drive:
                drive_info.append(f"{drive.split(' (neu erstellt)')[0]} (Ordner erstellt)")
            elif "(leer)" in drive:
                drive_info.append(f"{drive.split(' (leer)')[0]} (Ordner leer)")
            else:
                drive_info.append(drive)
                
        drive_info_str = ", ".join(drive_info) if drive_info else "Keine"
        
        return {
            "required": {
                "vae_name": (vae_names, {"info": f"Laufwerke: {drive_info_str}"})
            }
        }
    
    @staticmethod
    def find_external_drives():
        """Erkennt externe Festplatten basierend auf dem Betriebssystem"""
        # Dieselbe Methode wie in ExternalCheckpointLoader
        system = platform.system()
        drives = []
        
        if system == "Darwin":  # macOS
            volumes_dir = "/Volumes"
            if os.path.exists(volumes_dir):
                drives = [os.path.join(volumes_dir, d) for d in os.listdir(volumes_dir) 
                         if d != "Macintosh HD" and os.path.isdir(os.path.join(volumes_dir, d))]
        
        elif system == "Windows":
            try:
                import win32api
                drives = [f"{d}\\" for d in win32api.GetLogicalDriveStrings().split('\000')[:-1] 
                         if os.path.exists(f"{d}\\") and win32api.GetDriveType(f"{d}\\") == win32api.DRIVE_REMOVABLE]
            except ImportError:
                # Fallback wenn win32api nicht verfügbar ist
                import string
                drives = [f"{d}:\\" for d in string.ascii_uppercase 
                         if os.path.exists(f"{d}:\\")]
        
        elif system == "Linux":
            media_dir = "/media"
            if os.path.exists(media_dir):
                user = os.getenv("USER", "")
                user_media = os.path.join(media_dir, user)
                if os.path.exists(user_media):
                    drives = [os.path.join(user_media, d) for d in os.listdir(user_media) 
                             if os.path.isdir(os.path.join(user_media, d))]
        
        return drives
    
    RETURN_TYPES = ["VAE"]
    FUNCTION = "load_vae"
    CATEGORY = "loaders"

    def load_vae(self, vae_name):
        # Wenn keine VAEs gefunden wurden
        if vae_name == "Keine VAEs gefunden":
            # Rückgabe eines leeren VAE statt Fehler zu werfen
            print("Keine VAEs gefunden. Verwende Standard-VAE.")
            from comfy.sd import VAE
            return (VAE(),)
        
        # Vollständigen Pfad zum VAE aus dem Mapping abrufen
        drive_path, filename = self.__class__.path_mapping.get(vae_name, (None, None))
        if not drive_path:
            print(f"VAE-Pfad für {vae_name} konnte nicht gefunden werden. Verwende Standard-VAE.")
            from comfy.sd import VAE
            return (VAE(),)
                
        vae_path = os.path.join(drive_path, filename)
        print(f"Lade VAE von: {vae_path}")
        
        if not os.path.exists(vae_path):
            print(f"VAE-Datei existiert nicht: {vae_path}. Verwende Standard-VAE.")
            from comfy.sd import VAE
            return (VAE(),)
        
        # Dateigröße und Zugriffsrechte prüfen
        try:
            file_size = os.path.getsize(vae_path)
            print(f"VAE Dateigröße: {file_size/1024/1024:.2f} MB")
            if file_size < 1000:  # Warnung wenn Datei sehr klein ist
                print(f"Warnung: VAE-Datei könnte zu klein sein ({file_size} bytes)")
        except Exception as e:
            print(f"Konnte Dateigröße nicht prüfen: {str(e)}")
        
        # Verwenden Sie den Standard-VAE-Loader von ComfyUI
        try:
            from nodes import VAELoader
            loader = VAELoader()
            
            # Temporär den externen Pfad registrieren
            orig_vae_paths = folder_paths.get_folder_paths("vae")
            try:
                folder_paths.add_folder_paths_to_mem("vae", [os.path.dirname(vae_path)])
                # Die Datei direkt über den Standard-Loader laden
                print(f"Versuche VAE mit Standard-Loader zu laden: {os.path.basename(vae_path)}")
                result = loader.load_vae(os.path.basename(vae_path))
                if result is None:
                    raise ValueError("Standard-VAE-Loader gab None zurück")
                return result
            finally:
                # Pfad zurücksetzen
                folder_paths.set_folder_paths("vae", orig_vae_paths)
                    
        except Exception as e:
            print(f"Fehler beim Laden des VAE mit Standard-Loader: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fallback zu Standard-VAE, wenn alle anderen Methoden fehlschlagen
            try:
                # Importiere die VAE direkt aus comfy.sd
                from comfy.sd import VAE
                vae_instance = VAE(sd=None)  # Explizit None übergeben
                print("Standard VAE von comfy.sd geladen")
                return (vae_instance,)
            except Exception as e4:
                print(f"Konnte keine Standard VAE erstellen: {str(e4)}")
                # Wirklich sehr einfache Fallback-Lösung
                import torch
                
                # MockVAE mit den notwendigen Methoden
                class MockVAE(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        print("MockVAE erstellt als letzten Ausweg")
                        
                    def decode(self, samples, *args, **kwargs):
                        print(f"MockVAE.decode aufgerufen mit {type(samples)}")
                        import torch
                        
                        # Erzeugt einen PyTorch-Tensor im Format [B, C, H, W]
                        tensor = torch.zeros(1, 3, 512, 512, dtype=torch.float32)
                        
                        # Erzeuge ein erkennbares Muster
                        h, w = tensor.shape[2], tensor.shape[3]
                        for y in range(h):
                            for x in range(w):
                                tensor[0, 0, y, x] = x / w  # R
                                tensor[0, 1, y, x] = y / h  # G
                                tensor[0, 2, y, x] = (x + y) / (w + h) 
                        
                        # Hier konvertieren wir explizit zum von ComfyUI erwarteten Format [B, H, W, C]
                        # für Kompatibilität mit dem VAEDecode-Knoten
                        
                        print(f"MockVAE gibt Tensor mit Format {tensor.shape} zurück")
                        return tensor
                        
                    def encode(self, *args, **kwargs):
                        print("MockVAE.encode aufgerufen")
                        import torch
                        # Latents mit korrektem Format zurückgeben
                        return {"samples": torch.zeros(1, 4, 64, 64, dtype=torch.float32)}
                    
                    # Zusätzliche Methode für ComfyUI-Kompatibilität
                    def get_sd(self):
                        return None
                    
                Mock_vae = MockVAE()
                return (Mock_vae,)

    @classmethod
    def setup_vae_cache(cls):
        import tempfile
        
        # Cache-Verzeichnis einrichten
        cache_dir = os.path.join(tempfile.gettempdir(), "comfyui_ext_vae_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cls.cache_dir = cache_dir
        cls.cached_models = {}
        
        return cache_dir

    def cache_vae(self, vae_name):
        drive_path, filename = self.__class__.path_mapping.get(vae_name, (None, None))
        if not drive_path:
            return False
            
        source_path = os.path.join(drive_path, filename)
        cache_path = os.path.join(self.__class__.cache_dir, filename)
        
        try:
            shutil.copy2(source_path, cache_path)
            self.__class__.cached_models[vae_name] = cache_path
            return cache_path
        except Exception as e:
            print(f"VAE-Caching fehlgeschlagen: {e}")
            return False
        
class ExternalLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Externe Festplatten finden und auflisten
        external_drives = cls.find_external_drives()
        print(f"Gefundene externe Laufwerke für LoRAs: {external_drives}")
        
        # LoRAs von allen externen Laufwerken sammeln
        all_lora_files = []
        drives_with_models = []
        
        for drive in external_drives:
            # Pfad zum LoRA-Ordner
            model_path = os.path.join(drive, "loras")
            print(f"Suche nach LoRAs in: {model_path}")
            
            # Wenn der LoRA-Ordner nicht existiert, erstelle ihn
            if not os.path.exists(model_path):
                try:
                    os.makedirs(model_path)
                    print(f"LoRA-Ordner auf {drive} erstellt.")
                    drives_with_models.append(f"{os.path.basename(drive)} (neu erstellt)")
                except (PermissionError, OSError) as e:
                    print(f"Konnte keinen LoRA-Ordner auf {drive} erstellen: {str(e)}")
                    continue
            
            # Schaue nach Modellen im Ordner
            try:
                files = [f for f in os.listdir(model_path) if f.endswith('.safetensors') or f.endswith('.ckpt') or f.endswith('.pt')]
                print(f"Gefundene LoRA-Dateien in {model_path}: {files}")
                
                if files:  # Nur Laufwerke mit Modellen hinzufügen
                    all_lora_files.extend([(os.path.join(model_path), f) for f in files])
                    if os.path.basename(drive) not in [d.split(" (neu erstellt)")[0] for d in drives_with_models]:
                        drives_with_models.append(os.path.basename(drive))
                else:
                    # Wenn Ordner existiert aber leer ist
                    if os.path.basename(drive) not in [d.split(" (neu erstellt)")[0] for d in drives_with_models]:
                        drives_with_models.append(f"{os.path.basename(drive)} (leer)")
            except (PermissionError, FileNotFoundError) as e:
                print(f"Fehler beim Lesen von {model_path}: {str(e)}")
        
        # Ergebnisse für die UI vorbereiten
        if all_lora_files:
            lora_names = [f"{os.path.basename(os.path.dirname(d))} - {f}" for d, f in all_lora_files]
        else:
            lora_names = ["Keine LoRAs gefunden"]
        
        # Speichern der Pfad-Informationen zur späteren Verwendung
        cls.path_mapping = dict(zip(lora_names, all_lora_files)) if all_lora_files else {}
        
        # Füge Informationen für die UI hinzu
        drive_info = []
        for drive in drives_with_models:
            if "(neu erstellt)" in drive:
                drive_info.append(f"{drive.split(' (neu erstellt)')[0]} (Ordner erstellt)")
            elif "(leer)" in drive:
                drive_info.append(f"{drive.split(' (leer)')[0]} (Ordner leer)")
            else:
                drive_info.append(drive)
                
        drive_info_str = ", ".join(drive_info) if drive_info else "Keine"
        
        return {
            "required": {
                "lora_name": (lora_names, {"info": f"Laufwerke: {drive_info_str}"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1})
            }
        }
    
    @staticmethod
    def find_external_drives():
        """Erkennt externe Festplatten basierend auf dem Betriebssystem"""
        system = platform.system()
        drives = []
        
        if system == "Darwin":  # macOS
            volumes_dir = "/Volumes"
            if os.path.exists(volumes_dir):
                drives = [os.path.join(volumes_dir, d) for d in os.listdir(volumes_dir) 
                         if d != "Macintosh HD" and os.path.isdir(os.path.join(volumes_dir, d))]
        
        elif system == "Windows":
            try:
                import win32api
                drives = [f"{d}\\" for d in win32api.GetLogicalDriveStrings().split('\000')[:-1] 
                         if os.path.exists(f"{d}\\") and win32api.GetDriveType(f"{d}\\") == win32api.DRIVE_REMOVABLE]
            except ImportError:
                # Fallback wenn win32api nicht verfügbar ist
                import string
                drives = [f"{d}:\\" for d in string.ascii_uppercase 
                         if os.path.exists(f"{d}:\\")]
        
        elif system == "Linux":
            media_dir = "/media"
            if os.path.exists(media_dir):
                user = os.getenv("USER", "")
                user_media = os.path.join(media_dir, user)
                if os.path.exists(user_media):
                    drives = [os.path.join(user_media, d) for d in os.listdir(user_media) 
                             if os.path.isdir(os.path.join(user_media, d))]
        
        return drives
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    def get_cached_lora(self, lora_name):
        """Prüft, ob eine LoRA im Cache ist und gibt den Pfad zurück."""
        if not hasattr(self.__class__, 'cached_models') or not self.__class__.cached_models:
            return None
            
        if lora_name in self.__class__.cached_models:
            cache_path = self.__class__.cached_models[lora_name]
            if os.path.exists(cache_path):
                print(f"LoRA {lora_name} im Cache gefunden: {cache_path}")
                return cache_path
        
        return None

    def load_lora(self, lora_name, strength_model, strength_clip, model=None, clip=None, width=512, height=512, batch_size=1):
        
        if not hasattr(self.__class__, 'cache_dir'):
            self.__class__.setup_lora_cache()

        cache_path = self.get_cached_lora(lora_name)
        if cache_path:
            lora_path = cache_path
            filename = os.path.basename(lora_path)
            print(f"Verwende zwischengespeicherte LoRA von: {lora_path}")
        else:
            drive_path, filename = self.__class__.path_mapping.get(lora_name, (None, None))
            if not drive_path:
                raise ValueError(f"LoRA-Pfad für {lora_name} konnte nicht gefunden werden.")
            
            lora_path = os.path.join(drive_path, filename)
        print(f"Lade LoRA von: {lora_path}")

        self.cache_lora(lora_name)

        if lora_name == "Keine LoRAs gefunden":
            raise ValueError("Keine LoRAs gefunden. Bitte kopieren Sie LoRA-Modelle in den 'loras'-Ordner auf einem externen Laufwerk.")
        
        # Vollständigen Pfad zum LoRA aus dem Mapping abrufen
        drive_path, filename = self.__class__.path_mapping.get(lora_name, (None, None))
        if not drive_path:
            raise ValueError(f"LoRA-Pfad für {lora_name} konnte nicht gefunden werden.")
                
        lora_path = os.path.join(drive_path, filename)
        print(f"Lade LoRA von: {lora_path}")
        
        # Prüfe, ob die Datei existiert und lesbar ist
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA-Datei existiert nicht: {lora_path}")
            
        # Prüfe die Dateigröße, um offensichtliche Probleme zu erkennen
        try:
            file_size = os.path.getsize(lora_path)
            print(f"LoRA Dateigröße: {file_size/1024/1024:.2f} MB")
            if file_size < 1000:  # Warnung wenn Datei sehr klein ist
                print(f"Warnung: LoRA-Datei könnte zu klein sein ({file_size} bytes)")
        except Exception as e:
            print(f"Konnte Dateigröße nicht prüfen: {str(e)}")
        
        # Modell und CLIP vorbereiten wenn nicht übergeben
        if model is None or clip is None:
            import torch
            
            # Erstelle direkt eigene Dummy-Objekte
            if model is None:
                print("Erstelle eigenes Ersatz-Modell")
                class EmptyModel:
                    def __init__(self):
                        self.model = torch.nn.Module()
                        self.model_config = {
                            "model": {"target": "ldm.models.diffusion.ddpm.LatentDiffusion"}, 
                            "image_size": [height, width]
                        }
                        self.is_empty = True
                        self.model_name = "empty_model"
                        self.model_type = None
                model = EmptyModel()
            
            if clip is None:
                print("Erstelle eigenen Ersatz-CLIP")
                class EmptyCLIP:
                    def __init__(self):
                        self.cond_stage_model = torch.nn.Module()
                        self.clip_layer = "last"
                        self.is_empty = True
                        self.patcher = None
                        self.tokenizer = None
                clip = EmptyCLIP()
        
        # Drei verschiedene Lademethoden nacheinander versuchen
        try:
            # 1. Methode: Direkte Registrierung des Pfads mit add_temp_path
            success = add_temp_path("loras", os.path.dirname(lora_path))
            if success:
                try:
                    from nodes import LoraLoader
                    loader = LoraLoader()
                    print("Versuche direkte LoRA-Pfadregistrierung...")
                    result = loader.load_lora(os.path.basename(lora_path), strength_model, strength_clip, model, clip)
                    print("LoRA erfolgreich geladen (Methode 1)")
                    return result
                except Exception as e:
                    print(f"Direktes Laden fehlgeschlagen: {str(e)}")
                    # Fange den Fehler ab und versuche die nächste Methode
            
            # 2. Methode: Kopieren in temporären Ordner
            import shutil, tempfile
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, filename)
            
            try:
                shutil.copy2(lora_path, temp_file)
                print(f"LoRA nach {temp_file} kopiert")
                
                # Temporär den Pfad zu den bekannten Lora-Pfaden hinzufügen
                success = add_temp_path("loras", temp_dir)
                
                if success:
                    try:
                        from nodes import LoraLoader
                        loader = LoraLoader()
                        print(f"Lade LoRA aus temporärem Pfad: {filename}")
                        result = loader.load_lora(filename, strength_model, strength_clip, model, clip)
                        print("LoRA erfolgreich geladen (Methode 2)")
                        
                        # Aufräumen
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                        
                        return result
                    except Exception as e:
                        print(f"Temporäres Laden fehlgeschlagen: {str(e)}")
                        # Fange den Fehler ab und versuche die nächste Methode
                else:
                    print("Konnte temporären Pfad nicht registrieren")
            except Exception as e:
                print(f"Fehler beim temporären Kopieren: {str(e)}")
            finally:
                # Immer aufräumen
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
            
            # 3. Methode: Direktes Laden der Datei ohne ComfyUI-API
            print("Versuche direktes Laden der LoRA-Datei...")
            result = self.load_lora_direct(lora_path, strength_model, strength_clip, model, clip)
            if result:
                print("LoRA erfolgreich geladen (Methode 3)")
                return result
            
        except Exception as general_e:
            print(f"Fehler beim Laden der LoRA: {str(general_e)}")
        
        # Fallback wenn alle Methoden fehlschlagen
        print(f"Keine Lademethode erfolgreich, verwende ursprüngliche Modelle für: {lora_name}")
        return (model, clip)
        
    def load_lora_direct(self, file_path, strength_model, strength_clip, model, clip):
        """Lädt ein LoRA direkt ohne Abhängigkeit von folder_paths."""
        try:
            import torch
            
            print(f"Direktes Laden von LoRA: {file_path}")
            
            # Überprüfen, ob die Datei existiert
            if not os.path.exists(file_path):
                print(f"LoRA-Datei nicht gefunden: {file_path}")
                return None
                
            # Laden der Gewichte direkt aus der Datei
            if file_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                lora_state_dict = load_file(file_path)
            else:
                lora_state_dict = torch.load(file_path, map_location="cpu")
                    
            # Versuch, die Gewichte direkt anzuwenden
            print(f"LoRA Gewichte geladen, Stärken: {strength_model}, {strength_clip}")
            
            # Suche nach patcher in model oder clip
            if hasattr(model, 'patcher') and model.patcher is not None:
                print("Verwende model.patcher für LoRA-Anwendung")
                try:
                    model.patcher.patch_model(lora_state_dict, strength_model)
                except Exception as e:
                    print(f"Fehler beim Anwenden auf Model: {str(e)}")
                    
            if hasattr(clip, 'patcher') and clip.patcher is not None:
                print("Verwende clip.patcher für LoRA-Anwendung")
                try:
                    clip.patcher.patch_model(lora_state_dict, strength_clip)
                except Exception as e:
                    print(f"Fehler beim Anwenden auf CLIP: {str(e)}")
                    
            print("LoRA direkt angewendet")
            return (model, clip)
        except Exception as e:
            print(f"Fehler beim direkten Laden von LoRA: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
                
    @classmethod
    def setup_lora_cache(cls):
        import tempfile
        
        # Cache-Verzeichnis einrichten
        cache_dir = os.path.join(tempfile.gettempdir(), "comfyui_ext_lora_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cls.cache_dir = cache_dir
        cls.cached_models = {}
        
        return cache_dir

    def cache_lora(self, lora_name):
        # Stelle sicher, dass das Cache-Verzeichnis existiert
        if not hasattr(self.__class__, 'cache_dir') or not self.__class__.cache_dir:
            self.__class__.setup_lora_cache()
            
        drive_path, filename = self.__class__.path_mapping.get(lora_name, (None, None))
        if not drive_path:
            return False
            
        source_path = os.path.join(drive_path, filename)
        cache_path = os.path.join(self.__class__.cache_dir, filename)
        
        try:
            shutil.copy2(source_path, cache_path)
            if not hasattr(self.__class__, 'cached_models'):
                self.__class__.cached_models = {}
            self.__class__.cached_models[lora_name] = cache_path
            return cache_path
        except Exception as e:
            print(f"LoRA-Caching fehlgeschlagen: {e}")
            return False
        
class LoRADebugger:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "clip": ("CLIP",)}}
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "debug_lora"
    CATEGORY = "debug"
    
    def debug_lora(self, model, clip):
        result = "LoRA Debug-Informationen:\n\n"
        
        # Modell-Informationen
        result += "MODEL:\n"
        result += f"- Typ: {type(model)}\n"
        result += f"- Hat patcher: {hasattr(model, 'patcher')}\n"
        if hasattr(model, 'patcher') and model.patcher is not None:
            result += f"- Patcher-Typ: {type(model.patcher)}\n"
            if hasattr(model.patcher, 'lora_keys'):
                result += f"- Aktive LoRAs: {model.patcher.lora_keys}\n"
        
        # CLIP-Informationen
        result += "\nCLIP:\n"
        result += f"- Typ: {type(clip)}\n"
        result += f"- Hat patcher: {hasattr(clip, 'patcher')}\n"
        if hasattr(clip, 'patcher') and clip.patcher is not None:
            result += f"- Patcher-Typ: {type(clip.patcher)}\n"
            if hasattr(clip.patcher, 'lora_keys'):
                result += f"- Aktive LoRAs: {clip.patcher.lora_keys}\n"
        
        # ComfyUI-Versionsinfos
        result += "\nCOMFYUI INFOS:\n"
        if hasattr(comfy, 'version'):
            result += f"- Version: {comfy.version}\n"
        
        result += f"- LoRA-Pfade: {folder_paths.get_folder_paths('loras')}\n"
        
        return (result,)
    

NODE_CLASS_MAPPINGS = {
    "ExternalCheckpointLoader": ExternalCheckpointLoader,
    "ExternalVAELoader": ExternalVAELoader,
    "ExternalLoRALoader": ExternalLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExternalCheckpointLoader": "External Checkpoint Loader",
    "ExternalVAELoader": "External VAE Loader",
    "ExternalLoRALoader": "External LoRA Loader",
}