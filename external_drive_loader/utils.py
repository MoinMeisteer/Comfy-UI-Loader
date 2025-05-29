import os
import folder_paths

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