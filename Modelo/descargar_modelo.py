from huggingface_hub import snapshot_download
from pathlib import Path

# Carpeta raíz del laboratorio.
PROYECTO = Path(r"C:\LLM-Lab")

MODELO = PROYECTO / "02_modelo" / "SmolLM2-135M"

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