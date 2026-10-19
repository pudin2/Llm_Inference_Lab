import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import snapshot_download

from Utilidades.lab_comun import MODEL_PATH

MODELO = MODEL_PATH

MODELO.mkdir(parents=True, exist_ok=True)

print("Descargando modelo...")

ruta = snapshot_download(
    repo_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    local_dir=str(MODELO)
)

print("Modelo descargado en:")
print(ruta)

print("\nArchivos disponibles:")

for archivo in MODELO.iterdir():
    print(archivo.name)