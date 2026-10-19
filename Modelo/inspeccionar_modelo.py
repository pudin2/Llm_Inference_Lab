import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from Utilidades.lab_comun import cargar_modelo, leer_prompt

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

texto = leer_prompt("Inspección del modelo")

print("=== LABORATORIO LLM ===")

# ==========================================
# 2. CARGAR TOKENIZADOR Y MODELO
# ==========================================

tokenizer, model = cargar_modelo()

print("Tokenizador y modelo cargados.")

# ==========================================
# 4. INFORMACIÓN DEL MODELO
# ==========================================

print("\n=== ARQUITECTURA ===")

print("Tipo:", model.config.model_type)
print("Capas:", model.config.num_hidden_layers)
print("Dimensión oculta:", model.config.hidden_size)

total = sum(p.numel() for p in model.parameters())

print("Parámetros:", total)

# ==========================================
# 5. TOKENIZACIÓN
# ==========================================

inputs = tokenizer(
    texto,
    return_tensors="pt"
)

print("\n=== TOKENS ===")

print("IDs:", inputs["input_ids"])

print("Número de tokens:", inputs["input_ids"].shape[1])

# ==========================================
# 6. INFERENCIA
# ==========================================

print("\nEjecutando inferencia...")

with torch.inference_mode():

    outputs = model(
        **inputs,
        use_cache=False
    )

# ==========================================
# 7. INSPECCIONAR RESULTADO
# ==========================================

logits = outputs.logits

print("\n=== RESULTADOS ===")

print("Dimensiones logits:", logits.shape)

# Seleccionar el siguiente token
next_token = torch.argmax(
    logits[0, -1, :]
).item()

print("Siguiente token ID:", next_token)

print(
    "Token decodificado:",
    repr(tokenizer.decode([next_token]))
)

print("\nINFERENCIA COMPLETADA")