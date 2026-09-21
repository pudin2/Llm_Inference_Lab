import torch
import transformers
import safetensors
import sentencepiece

print("=== VERIFICACIÓN DEL ENTORNO ===")

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Safetensors:", safetensors.__version__)
print("SentencePiece:", sentencepiece.__version__)

print("\nTodas las dependencias funcionan correctamente.")