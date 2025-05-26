import os
import platform
import folder_paths
import comfy
import shutil

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
        
        # Verwenden Sie den Standard-Checkpoint-Loader von ComfyUI
        try:
            from nodes import CheckpointLoaderSimple
            loader = CheckpointLoaderSimple()
            
            # Temporär den externen Pfad registrieren
            orig_checkpoints = folder_paths.get_folder_paths("checkpoints")
            try:
                folder_paths.add_folder_paths_to_mem("checkpoints", [os.path.dirname(ckpt_path)])
                # Die Datei direkt über den Standard-Loader laden
                return loader.load_checkpoint(os.path.basename(ckpt_path))
            finally:
                # Pfad zurücksetzen
                folder_paths.set_folder_paths("checkpoints", orig_checkpoints)
                
        except Exception as e:
            print(f"Fehler beim Laden des Checkpoints: {str(e)}")
            # Fallback-Methode
            try:
                from comfy import sd
                result = sd.load_checkpoint_guess_config(ckpt_path)
                # Stellen Sie sicher, dass wir genau 3 Elemente zurückgeben
                if len(result) >= 3:
                    return (result[0], result[1], result[2])
                else:
                    raise ValueError(f"Unerwartetes Format des Checkpoint-Ergebnisses")
            except Exception as e2:
                print(f"Auch Fallback fehlgeschlagen: {str(e2)}")
                raise ValueError(f"Konnte Checkpoint nicht laden: {ckpt_name}")
    

NODE_CLASS_MAPPINGS = {
    "ExternalCheckpointLoader": ExternalCheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExternalCheckpointLoader": "External Checkpoint Loader",   
}