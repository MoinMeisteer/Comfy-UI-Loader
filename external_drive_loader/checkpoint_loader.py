import os
import platform
import shutil
import folder_paths
from .utils import add_temp_path

class ExternalCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Externe Festplatten finden und auflisten
        external_drives = cls.find_external_drives()
        print(f"Gefundene externe Laufwerke: {external_drives}")
        
        # Checkpoints von allen externen Laufwerken sammeln
        all_ckpt_files = []
        drives_with_models = []
        drives_without_folders = []
        
        for drive in external_drives:
            # Pfad zum Checkpoints-Ordner
            model_path = os.path.join(drive, "checkpoints")
            print(f"Suche nach Checkpoints in: {model_path}")
            
            # Prüfe, ob der Ordner existiert, aber erstelle ihn nicht automatisch
            if not os.path.exists(model_path):
                print(f"Checkpoints-Ordner auf {drive} nicht gefunden.")
                drives_without_folders.append(os.path.basename(drive))
                continue
            
            # Schaue nach Modellen im Ordner
            try:
                files = [f for f in os.listdir(model_path) if f.endswith('.safetensors') or f.endswith('.ckpt')]
                print(f"Gefundene Dateien in {model_path}: {files}")
                
                if files:  # Nur Laufwerke mit Modellen hinzufügen
                    all_ckpt_files.extend([(os.path.join(model_path), f) for f in files])
                    drives_with_models.append(os.path.basename(drive))
                else:
                    # Ordner existiert aber ist leer
                    drives_with_models.append(f"{os.path.basename(drive)} (leer)")
            except (PermissionError, FileNotFoundError) as e:
                print(f"Fehler beim Lesen von {model_path}: {str(e)}")
        
        # Vorbereiten der Beschriftungen für die UI
        if drives_without_folders:
            missing_drives = ", ".join(drives_without_folders)
            drives_without_folders_text = f"Auf folgenden Laufwerken fehlt der 'checkpoints' Ordner: {missing_drives}"
        else:
            drives_without_folders_text = ""
        
        # Ergebnisse für die UI vorbereiten
        if all_ckpt_files:
            # Es wurden Modelle gefunden
            ckpt_names = [f"{os.path.basename(os.path.dirname(d))} - {f}" for d, f in all_ckpt_files]
            
            # Erstelle Info-Text mit fehlenden Ordnern
            drive_info = []
            for drive in drives_with_models:
                if "(leer)" in drive:
                    drive_info.append(f"{drive.split(' (leer)')[0]} (Ordner leer)")
                else:
                    drive_info.append(drive)
                    
            info_text = f"Laufwerke mit Modellen: {', '.join(drive_info)}"
            if drives_without_folders_text:
                info_text += f"\n\n{drives_without_folders_text}"
        else:
            # Keine Modelle gefunden - zeige klare Anleitung
            if drives_without_folders:
                # Es gibt Laufwerke ohne checkpoints Ordner
                ckpt_names = ["Bitte Modelle in 'checkpoints' Ordner kopieren"]
                info_text = f"Keine Modelle gefunden.\n\n{drives_without_folders_text}\n\nBitte erstellen Sie einen 'checkpoints' Ordner und kopieren Sie dort Ihre Modelle hinein."
            else:
                # Es gibt Laufwerke mit leeren checkpoints Ordnern
                ckpt_names = ["Bitte Modelle in 'checkpoints' Ordner kopieren"]
                info_text = "Keine Modelle gefunden. Bitte kopieren Sie Modelle in den vorhandenen 'checkpoints' Ordner."
        
        # Speichern der Pfad-Informationen zur späteren Verwendung
        cls.path_mapping = dict(zip(ckpt_names, all_ckpt_files)) if all_ckpt_files else {}
        
        # UI-Erstellung mit Button zum Ordner erstellen
        ui_dict = {
            "required": {
                "ckpt_name": (ckpt_names, {"info": info_text})
            }
        }
        
        # Füge Button zum Erstellen des Ordners hinzu, aber nur wenn es fehlende Ordner gibt
        if drives_without_folders:
            ui_dict["optional"] = {
                "create_checkpoints_folder": ("BOOLEAN", {
                    "default": False, 
                    "label": "Checkpoints-Ordner erstellen"
                })
            }
        
        return ui_dict
    
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

    def load_ckpt(self, ckpt_name, create_folders=False, create_checkpoints_folder=False):
        """Lädt einen Checkpoint von einem externen Laufwerk."""
        
        # Wenn keine Modelle gefunden wurden aber der Button gedrückt wurde
        if "Bitte Modelle" in ckpt_name and create_checkpoints_folder == True:
            # Externe Festplatten erneut finden
            external_drives = self.__class__.find_external_drives()
            created_folders = []
            
            for drive in external_drives:
                model_path = os.path.join(drive, "checkpoints")
                if not os.path.exists(model_path):
                    try:
                        os.makedirs(model_path)
                        created_folders.append(os.path.basename(drive))
                    except (PermissionError, OSError) as e:
                        print(f"Konnte keinen Checkpoints-Ordner auf {drive} erstellen: {str(e)}")
            
            if created_folders:
                msg = f"Checkpoints-Ordner wurden erstellt auf: {', '.join(created_folders)}"
                print(msg)
                # Wir geben ein leeres Modell zurück
                from comfy.sd import ModelPatcher, CLIP, VAE
                model = ModelPatcher()
                clip = CLIP()
                vae = VAE()
                return (model, clip, vae)
        
        # Wenn keine Modelle gefunden wurden
        if "Bitte Modelle" in ckpt_name:
            # Unterscheide zwischen "Ordner fehlt" und "Ordner ist leer"
            error_msg = "Keine Modelle gefunden."
            
            # Wenn wir wissen, dass es Laufwerke ohne Checkpoints-Ordner gibt
            if hasattr(self.__class__, 'drives_without_folders') and self.__class__.drives_without_folders:
                missing_drives = ", ".join(self.__class__.drives_without_folders)
                error_msg += f"\n\nAuf folgenden Laufwerken fehlt der 'checkpoints' Ordner: {missing_drives}"
                
                # Zeige an, wie der Benutzer den Ordner erstellen kann
                if create_checkpoints_folder is False:  # Nur wenn der Button nicht aktiv ist
                    error_msg += "\n\nOption 1: Aktivieren Sie 'Checkpoints-Ordner erstellen' und führen Sie diesen Knoten erneut aus."
                    
                error_msg += "\nOption 2: Erstellen Sie manuell einen 'checkpoints' Ordner auf Ihrem externen Laufwerk."
                error_msg += "\nOption 3: Kopieren Sie Ihre .safetensors oder .ckpt Dateien in einen vorhandenen 'checkpoints' Ordner."
                
            else:
                # Wenn es Ordner gibt, die aber leer sind
                error_msg += " Bitte kopieren Sie Ihre .safetensors oder .ckpt Dateien in den vorhandenen 'checkpoints' Ordner."
            
            raise ValueError(error_msg)
        
        # Stelle sicher dass das Cache-Verzeichnis existiert
        if not hasattr(self.__class__, 'cache_dir'):
            self.__class__.setup_model_cache()
        
        # Pfad zum Checkpoint-Modell bestimmen (entweder aus Cache oder original)
        cache_path = self.get_cached_checkpoint(ckpt_name)
        if cache_path:
            ckpt_path = cache_path
            print(f"Verwende zwischengespeicherten Checkpoint: {ckpt_path}")
        else:
            # Vollständigen Pfad zum Checkpoint aus dem Mapping abrufen
            drive_path, filename = self.__class__.path_mapping.get(ckpt_name, (None, None))
            if not drive_path:
                raise ValueError(f"Modellpfad für {ckpt_name} konnte nicht gefunden werden.")
                    
            ckpt_path = os.path.join(drive_path, filename)
            print(f"Lade Checkpoint von: {ckpt_path}")
            
            # Im Hintergrund cachen für zukünftige Verwendung
            self.cache_checkpoint(ckpt_name)
        
        # Überprüfe, ob die Datei tatsächlich existiert
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Die Checkpoint-Datei {ckpt_path} existiert nicht oder ist nicht zugänglich.")
            
        # Überprüfe die Dateigröße
        try:
            file_size = os.path.getsize(ckpt_path) / (1024 * 1024)  # in MB
            print(f"Checkpoint-Größe: {file_size:.2f} MB")
            if file_size < 100:
                print(f"Warnung: Die Datei ist möglicherweise zu klein für ein vollständiges Modell ({file_size:.2f} MB)")
        except Exception as e:
            print(f"Konnte Dateigröße nicht prüfen: {str(e)}")
        
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
            raise ValueError(f"Konnte Checkpoint nicht laden: {ckpt_name}. Bitte überprüfen Sie, ob die Datei ein gültiges Modell ist.")
            
    @classmethod
    def setup_model_cache(cls):
        import tempfile
        
        # Cache-Verzeichnis einrichten
        cache_dir = os.path.join(tempfile.gettempdir(), "comfyui_ext_checkpoint_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cls.cache_dir = cache_dir
        cls.cached_models = {}
        
        return cache_dir

    def cache_checkpoint(self, ckpt_name):
        # Stelle sicher, dass das Cache-Verzeichnis existiert
        if not hasattr(self.__class__, 'cache_dir') or not self.__class__.cache_dir:
            self.__class__.setup_model_cache()
            
        drive_path, filename = self.__class__.path_mapping.get(ckpt_name, (None, None))
        if not drive_path:
            return False
            
        source_path = os.path.join(drive_path, filename)
        cache_path = os.path.join(self.__class__.cache_dir, filename)
        
        try:
            # Optional: Asynchrones Kopieren für große Dateien
            print(f"Kopiere Checkpoint {filename} in den Cache...")
            shutil.copy2(source_path, cache_path)
            
            if not hasattr(self.__class__, 'cached_models'):
                self.__class__.cached_models = {}
            self.__class__.cached_models[ckpt_name] = cache_path
            print(f"Checkpoint {filename} erfolgreich gecacht")
            return cache_path
        except Exception as e:
            print(f"Checkpoint-Caching fehlgeschlagen: {e}")
            return False
        
    def get_cached_checkpoint(self, ckpt_name):
        """Prüft, ob ein Checkpoint im Cache ist und gibt den Pfad zurück."""
        if not hasattr(self.__class__, 'cached_models') or not self.__class__.cached_models:
            return None
            
        if ckpt_name in self.__class__.cached_models:
            cache_path = self.__class__.cached_models[ckpt_name]
            if os.path.exists(cache_path):
                print(f"Checkpoint {ckpt_name} im Cache gefunden: {cache_path}")
                return cache_path
        
        return None