import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from Utilidades.lab_comun import cargar_modelo, leer_prompt

texto = leer_prompt("Experimento 01: instrumentación de Q")

# 1. Cargar modelo
tokenizer, model = cargar_modelo()

# 2. Seleccionar la primera proyección Q
q_proj = model.model.layers[0].self_attn.q_proj

print("=== MATRIZ Wq ===")
print("Dimensiones:", tuple(q_proj.weight.shape))

# 3. Registrar datos de la operación
registro = {}

def hook_q(module, inputs, output):

    X = inputs[0]

    W = module.weight

    Q = output

    registro["X"] = X.detach().clone()
    registro["W"] = W.detach().clone()
    registro["Q"] = Q.detach().clone()

    K = module.in_features
    N = module.out_features
    M = X.numel() // K

    registro["multiplicaciones"] = M * K * N

    registro["sumas"] = M * N * (K - 1)

    if module.bias is not None:
        registro["sumas"] += M * N

# 4. Instalar hook
handle = q_proj.register_forward_hook(hook_q)

# 5. Ejecutar inferencia
inputs = tokenizer(
    texto,
    return_tensors="pt"
)

with torch.inference_mode():
    outputs = model(**inputs, use_cache=False)

# 6. Retirar hook
handle.remove()

# 7. Mostrar resultados
print("\n=== TENSORES ===")

print("Entrada X:", registro["X"].shape)
print("Pesos W:", registro["W"].shape)
print("Salida Q:", registro["Q"].shape)

print("\n=== CONTEO ===")

print("Multiplicaciones:", registro["multiplicaciones"])
print("Sumas:", registro["sumas"])

print(
    "Total:",
    registro["multiplicaciones"] + registro["sumas"]
)

# 8. Validar la operación
Q_recalculado = torch.nn.functional.linear(
    registro["X"],
    registro["W"],
    q_proj.bias
)

print("\n=== VALIDACIÓN ===")

print(
    "¿Resultados equivalentes?:",
    torch.allclose(
        registro["Q"],
        Q_recalculado,
        rtol=1e-5,
        atol=1e-6
    )
)

print(
    "Diferencia máxima:",
    (registro["Q"] - Q_recalculado).abs().max().item()
)