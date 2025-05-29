import os
import platform
import shutil
import inspect
import folder_paths

class ExternalLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Externe Festplatten finden und auflisten
        external_drives = cls.find_external_drives()
        print(f"Gefundene externe Laufwerke für LoRAs: {external_drives}")
        
        # LoRAs von allen externen Laufwerken sammeln
        all_lora_files = []
        drives_with_models = []
        drives_without_folders = []
        
        for drive in external_drives:
            # Pfad zum LoRAs-Ordner
            model_path = os.path.join(drive, "loras")
            print(f"Suche nach LoRAs in: {model_path}")
            
            # Prüfe, ob der Ordner existiert
            if not os.path.exists(model_path):
                print(f"LoRAs-Ordner auf {drive} nicht gefunden.")
                drives_without_folders.append(os.path.basename(drive))
                continue
            
            # Schaue nach Modellen im Ordner
            try:
                files = [f for f in os.listdir(model_path) if f.endswith('.safetensors') or f.endswith('.pt')]
                print(f"Gefundene LoRA-Dateien in {model_path}: {files}")
                
                if files:  # Nur Laufwerke mit Modellen hinzufügen
                    all_lora_files.extend([(os.path.join(model_path), f) for f in files])
                    drives_with_models.append(os.path.basename(drive))
                else:
                    # Ordner existiert aber ist leer
                    drives_with_models.append(f"{os.path.basename(drive)} (leer)")
            except (PermissionError, FileNotFoundError) as e:
                print(f"Fehler beim Lesen von {model_path}: {str(e)}")
        
        # Fehlende Laufwerke-Information für die UI
        missing_folders_info = ""
        if drives_without_folders:
            missing_drives = ", ".join(drives_without_folders)
            missing_folders_info = f" (Erstellen Sie 'loras'-Ordner auf: {missing_drives})"
        
        # Ergebnisse für die UI vorbereiten
        if all_lora_files:
            lora_names = [f"{os.path.basename(os.path.dirname(d))} - {f}" for d, f in all_lora_files]
        else:
            # Anleitung hinzufügen, wenn keine Modelle gefunden wurden
            lora_names = [f"Keine LoRAs gefunden{missing_folders_info}"]
        
        # Speichern der Pfad-Informationen zur späteren Verwendung
        cls.path_mapping = dict(zip(lora_names, all_lora_files)) if all_lora_files else {}
        
        # Füge Informationen für die UI hinzu
        drive_info = []
        for drive in drives_with_models:
            if "(leer)" in drive:
                drive_info.append(f"{drive.split(' (leer)')[0]} (Ordner leer)")
            else:
                drive_info.append(drive)
        
        # Erstelle Info-Text mit fehlenden Ordnern
        info_text = f"Laufwerke: {', '.join(drive_info) if drive_info else 'Keine'}"
        if drives_without_folders:
            info_text += f"\nFehlende 'loras'-Ordner auf: {', '.join(drives_without_folders)}"
            info_text += "\nAktivieren Sie 'LoRA-Ordner erstellen', um sie anzulegen"
        
        # Erstelle UI mit optionalen Parametern
        ui_dict = {
            "required": {
                "lora_name": (lora_names, {"info": info_text}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",)
            }
        }
        
        # Optional: Füge Button zum Erstellen von Ordnern hinzu
        if drives_without_folders:
            ui_dict["optional"]["create_folders"] = ("BOOLEAN", {"default": False, "label": "LoRA-Ordner erstellen"})
        
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

    def load_lora(self, lora_name, strength_model, strength_clip, model=None, clip=None, create_folders=False):
        """Verbesserte LoRA-Ladefunktion mit Kompatibilität für verschiedene ComfyUI-Versionen."""
        # Prüfe, ob Ordner erstellt werden sollen
        if create_folders and "Keine LoRAs gefunden" in lora_name:
            # Externe Festplatten erneut finden
            external_drives = self.__class__.find_external_drives()
            created_folders = []
            
            for drive in external_drives:
                lora_path = os.path.join(drive, "loras")
                if not os.path.exists(lora_path):
                    try:
                        os.makedirs(lora_path)
                        created_folders.append(os.path.basename(drive))
                    except (PermissionError, OSError) as e:
                        print(f"Konnte keinen LoRA-Ordner auf {drive} erstellen: {str(e)}")
            
            if created_folders:
                msg = f"LoRA-Ordner wurden erstellt auf: {', '.join(created_folders)}"
                print(msg)
                # Gib unveränderte Modelle zurück
                return (model, clip)
        
        # Initialisiere das Cache-System
        if not hasattr(self.__class__, 'cache_dir'):
            self.__class__.setup_lora_cache()
        
        # Prüfe auf leere Auswahl
        if lora_name == "Keine LoRAs gefunden":
            raise ValueError("Keine LoRAs gefunden. Bitte kopieren Sie LoRA-Modelle in den 'loras'-Ordner auf einem externen Laufwerk oder aktivieren Sie 'LoRA-Ordner erstellen'.")
        
        # Bestimme den Dateipfad (entweder aus Cache oder vom externen Laufwerk)
        cache_path = self.get_cached_lora(lora_name)
        if cache_path and os.path.exists(cache_path):
            lora_path = cache_path
            filename = os.path.basename(lora_path)
            print(f"Verwende zwischengespeicherte LoRA: {lora_path}")
        else:
            # Wenn nicht im Cache, verwende das Original
            drive_path, filename = self.__class__.path_mapping.get(lora_name, (None, None))
            if not drive_path:
                raise ValueError(f"LoRA-Pfad für {lora_name} konnte nicht gefunden werden.")
                    
            lora_path = os.path.join(drive_path, filename)
            print(f"Lade LoRA von: {lora_path}")
            
            # Im Hintergrund cachen für zukünftige Verwendung
            try:
                self.cache_lora(lora_name)
            except Exception as e:
                print(f"Caching fehlgeschlagen (nicht kritisch): {e}")
        
        # Wichtig: Detaillierte Dateiprüfung
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA-Datei existiert nicht: {lora_path}")
        
        try:
            file_size = os.path.getsize(lora_path)
            print(f"LoRA Dateigröße: {file_size/1024/1024:.2f} MB")
            if file_size < 10000:  # Warnung wenn Datei sehr klein ist
                print(f"WARNUNG: LoRA-Datei ist sehr klein ({file_size} bytes), könnte beschädigt sein")
        except Exception as e:
            print(f"Konnte Dateigröße nicht prüfen: {str(e)}")
        
        # Erstelle leere Modelle falls nötig
        if model is None or clip is None:
            import torch
            from comfy import sd
            
            print("Erstelle leere Modelle für alleinstehende LoRA-Anwendung")
            if model is None:
                model = sd.load_model_weights_empty(512, 512, 1)
            if clip is None:
                clip = sd.load_clip_weights_empty(512, 512, 1)

        # METHODE 1: Verwende lora.load_lora_weights mit richtigen Parametern
        try:
            from comfy import lora
            from safetensors.torch import load_file
            from comfy.utils import load_torch_file
            
            print(f"Lade LoRA mit korrigierter Methode: {lora_path}")
            
            # Wähle die richtige Ladefunktion basierend auf der Dateiendung
            if lora_path.endswith(".safetensors"):
                print("Lade safetensors-Format...")
                lora_state_dict = load_file(lora_path)
            else:
                print("Lade pytorch-Format...")
                lora_state_dict = load_torch_file(lora_path)
                
            # Prüfe, welche lora.load_* Funktionen verfügbar sind und welche Parameter sie akzeptieren
            import inspect
            
            if hasattr(lora, 'load_lora_for_models'):
                print("Verwende load_lora_for_models Funktion")
                model_lora, clip_lora = lora.load_lora_for_models(model, clip, lora_state_dict, strength_model, strength_clip)
            elif hasattr(lora, 'load_lora'):
                sig = inspect.signature(lora.load_lora)
                param_count = len(sig.parameters)
                print(f"load_lora hat {param_count} Parameter")
                
                if param_count >= 5:  # Neuere Versionen akzeptieren alle Parameter
                    model_lora, clip_lora = lora.load_lora(model, clip, lora_state_dict, strength_model, strength_clip)
                elif param_count == 3:  # Einige Versionen erwarten lora_state_dict als dritten Parameter
                    try:
                        # In manchen 3-Parameter-Versionen können die Parameter anders angeordnet sein
                        # Versuche zunächst den Ansatz mit model, clip, lora_state_dict
                        print("Versuche load_lora mit model, clip, state_dict")
                        
                        # Prüfe, ob clip ein iterierbares Objekt sein sollte
                        if hasattr(lora, 'load_lora'):
                            sig = inspect.signature(lora.load_lora)
                            param_names = list(sig.parameters.keys())
                            print(f"Parameter-Namen: {param_names}")
                            
                            # Die richtige Art und Reihenfolge bestimmen
                            if 'to_load' in param_names:
                                print("Verwende load_lora mit korrekter Parameterreihenfolge")
                                # In dieser Version erwartet lora.load_lora lora_dict, model, strength
                                model_lora = model.clone() if hasattr(model, 'clone') else model
                                clip_lora = clip.clone() if hasattr(clip, 'clone') else clip
                                
                                # Versuche den manuellen Patch-Ansatz stattdessen
                                if hasattr(model_lora, 'add_patches'):
                                    print("Verwende add_patches für Model")
                                    model_lora.add_patches(lora_state_dict, strength_model)
                                
                                if hasattr(clip_lora, 'add_patches'):
                                    print("Verwende add_patches für CLIP")
                                    clip_lora.add_patches(lora_state_dict, strength_clip)
                                
                                # Alternative: Direktes patchen mit der apply_lora Methode wenn vorhanden
                                elif hasattr(lora, 'apply_lora'):
                                    print("Verwende apply_lora Methode")
                                    lora.apply_lora(model_lora, lora_state_dict, strength_model)
                                    lora.apply_lora(clip_lora, lora_state_dict, strength_clip)
                                
                                return (model_lora, clip_lora)
                            else:
                                # Versuchen wir den ursprünglichen Ansatz
                                model_lora, clip_lora = lora.load_lora(model, clip, lora_state_dict)
                        
                        # Stärken müssen separat angewendet werden
                        if hasattr(model_lora, 'set_lora_strength'):
                            model_lora.set_lora_strength(strength_model)
                        if hasattr(clip_lora, 'set_lora_strength'):
                            clip_lora.set_lora_strength(strength_clip)
                            
                    except TypeError as e:
                        print(f"TypeError bei 3-Parameter-Aufruf: {e}")
                        print("Versuche alternativen LoRA-Anwendungsansatz")
                        
                        # Versuche direkt auf das Modell und CLIP anzuwenden
                        model_lora = model.clone() if hasattr(model, 'clone') else model
                        clip_lora = clip.clone() if hasattr(clip, 'clone') else clip
                        
                        # Direktes Anwenden mit Patchern falls vorhanden
                        if hasattr(model_lora, 'add_patches'):
                            model_lora.add_patches(lora_state_dict, strength_model)
                        if hasattr(clip_lora, 'add_patches'):
                            clip_lora.add_patches(lora_state_dict, strength_clip)
                            
                        return (model_lora, clip_lora)
                    except Exception as e:
                        print(f"Fehler beim Anwenden des LoRA mit 3-Parameter-Ansatz: {e}")
                        raise e
                
            print("LoRA erfolgreich angewendet")
            return (model_lora, clip_lora)
    
        except Exception as e:
            print(f"Fehler beim direkten Laden der LoRA: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # METHODE 2: Versuche mit dem Standard LoraLoader
            try:
                from nodes import LoraLoader
                loader = LoraLoader()
                print("Versuche Standard LoraLoader als Fallback...")
                
                # Kopiere die Datei in den Standard-LoRA-Ordner
                lora_dirs = folder_paths.get_folder_paths("loras")
                if not lora_dirs:
                    raise ValueError("Keine Standard-LoRA-Ordner gefunden")
                
                base_filename = os.path.basename(lora_path)
                tmp_file = os.path.join(lora_dirs[0], f"tmp_{base_filename}")
                print(f"Kopiere {lora_path} nach {tmp_file}")
                
                shutil.copy2(lora_path, tmp_file)
                try:
                    tmp_filename = os.path.basename(tmp_file)
                    print(f"Rufe Standard-LoraLoader.load_lora mit {tmp_filename} auf")
                    
                    import inspect
                    sig = inspect.signature(loader.load_lora)
                    param_names = list(sig.parameters.keys())
                    param_count = len(param_names)
                    
                    print(f"LoraLoader.load_lora hat {param_count} Parameter: {param_names}")
                    
                    # KRITISCHE KORREKTUR: Verwenden Sie die korrekte Parameterreihenfolge
                    if "model" in param_names and "clip" in param_names and "lora_name" in param_names:
                        # Die Parameter sind in der Reihenfolge model, clip, lora_name, strength_model, strength_clip
                        print("Verwende korrigierte Parameterreihenfolge")
                        result = loader.load_lora(
                            model,                 # model
                            clip,                  # clip
                            str(tmp_filename),     # lora_name als String
                            float(strength_model), # strength_model als Float
                            float(strength_clip)   # strength_clip als Float
                        )
                        return result
                    elif param_count == 5:
                        # Fallback zu anderen möglichen Ordnungen, erst mit lora_name
                        try:
                            print("Versuche alternative Parameterreihenfolge (lora_name zuerst)")
                            result = loader.load_lora(
                                str(tmp_filename),     # lora_name
                                float(strength_model), # strength_model
                                float(strength_clip),  # strength_clip
                                model,                 # model
                                clip                   # clip
                            )
                            return result
                        except Exception as e:
                            print(f"Alternative Reihenfolge 1 fehlgeschlagen: {str(e)}")
                    
                    # Wenn nichts funktioniert, versuche die native Methode mit manuell geladenen Weights
                    print("Versuche manuelles Laden der LoRA-Gewichte")
                    if tmp_file.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        lora_state_dict = load_file(tmp_file)
                    else:
                        import torch
                        lora_state_dict = torch.load(tmp_file, map_location="cpu")
                    
                    # Ältere Versionen von ComfyUI verwenden eine andere API für LoRA
                    try:
                        from comfy import lora as comfy_lora
                        model_lora = model
                        clip_lora = clip
                        
                        # Durchlaufe alle Schlüssel im LoRA-Wörterbuch
                        a_tensors = {}
                        b_tensors = {}
                        
                        # Importiere die Patcher Klasse
                        try:
                            from comfy.lora import LoRAPatches
                            # Manuell patchen
                            if hasattr(model, 'lora_forward'):  # Direkte Anwendung des LoRA
                                print("Manuelles Patchen mit model.lora_forward")
                                model_lora = model.clone()
                                model_lora.lora_forward(lora_state_dict, float(strength_model))
                                
                            if hasattr(clip, 'lora_forward'):
                                print("Manuelles Patchen mit clip.lora_forward")
                                clip_lora = clip.clone()
                                clip_lora.lora_forward(lora_state_dict, float(strength_clip))
                                
                            return (model_lora, clip_lora)
                        except Exception as e:
                            print(f"Manuelles Patchen fehlgeschlagen: {str(e)}")
                    except Exception as e:
                        print(f"Natives Laden fehlgeschlagen: {str(e)}")
                    
                    # Als letzten Ausweg, gib die unveränderten Modelle zurück
                    print("Alle Versuche fehlgeschlagen, gebe unveränderte Modelle zurück")
                    return (model, clip)
                finally:
                    # Lösche die temporäre Datei
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                        print("Temporäre Datei gelöscht")
            
            except Exception as e2:
                print(f"Auch der Standard-LoraLoader ist fehlgeschlagen: {str(e2)}")
                traceback.print_exc()
        
        # Wenn alle Methoden fehlschlagen, gib die unveränderten Modelle zurück
        print("LoRA konnte nicht angewendet werden, gebe unveränderte Modelle zurück")
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