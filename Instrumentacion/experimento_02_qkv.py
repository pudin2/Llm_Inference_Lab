import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from Utilidades.lab_comun import cargar_modelo, leer_prompt
from Utilidades.prediccion import siguiente_token
from Utilidades.tensores import a_numpy

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

texto = leer_prompt("Experimento 02: instrumentación de Q, K y V")

tokenizer, model = cargar_modelo()

# ==========================================
# 2. LOCALIZAR LA PRIMERA CAPA
# ==========================================

attention = model.model.layers[0].self_attn

proyecciones = {
    "Q": attention.q_proj,
    "K": attention.k_proj,
    "V": attention.v_proj
}

registros = {}
handles = []

# ==========================================
# 3. CREAR INSTRUMENTACIÓN
# ==========================================

def crear_hook(nombre):

    def hook(module, inputs, output):

        X = inputs[0]

        W = module.weight

        M = X.numel() // module.in_features
        K = module.in_features
        N = module.out_features

        multiplicaciones = M * K * N

        sumas = M * N * (K - 1)

        if module.bias is not None:
            sumas += M * N

        registros[nombre] = {
            "entrada": X.detach().clone(),
            "pesos": W.detach().clone(),
            "salida": output.detach().clone(),
            "multiplicaciones": multiplicaciones,
            "sumas": sumas
        }

    return hook

# ==========================================
# 4. INSTALAR HOOKS
# ==========================================

for nombre, modulo in proyecciones.items():

    handle = modulo.register_forward_hook(
        crear_hook(nombre)
    )

    handles.append(handle)

# ==========================================
# 5. EJECUTAR INFERENCIA
# ==========================================

inputs = tokenizer(
    texto,
    return_tensors="pt"
)

try:
    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=False
        )

finally:
    for handle in handles:
        handle.remove()

# ==========================================
# 6. MOSTRAR Y VALIDAR RESULTADOS
# ==========================================

total_mult = 0
total_sumas = 0

for nombre, datos in registros.items():

    modulo = proyecciones[nombre]

    print(f"\n=== PROYECCIÓN {nombre} ===")

    print("Entrada:", tuple(datos["entrada"].shape))
    print("Pesos:", tuple(datos["pesos"].shape))
    print("Salida:", tuple(datos["salida"].shape))

    print("Multiplicaciones:", datos["multiplicaciones"])
    print("Sumas:", datos["sumas"])

    recalculado = F.linear(
        datos["entrada"],
        datos["pesos"],
        modulo.bias
    )

    diferencia = (
        datos["salida"] - recalculado
    ).abs().max().item()

    print(
        "Equivalencia:",
        torch.allclose(
            datos["salida"],
            recalculado,
            rtol=1e-5,
            atol=1e-6
        )
    )

    print("Diferencia máxima:", diferencia)

    total_mult += datos["multiplicaciones"]
    total_sumas += datos["sumas"]

# ==========================================
# 7. RESULTADO CONSOLIDADO
# ==========================================

print("\n=== TOTAL Q + K + V ===")

print("Multiplicaciones:", total_mult)
print("Sumas:", total_sumas)
print("Operaciones:", total_mult + total_sumas)

token_id, token_predicho = siguiente_token(tokenizer, outputs.logits)

print("\nEXPERIMENTO FINALIZADO")

# ==========================================
# 8. COPIAS EN NUMPY PARA EL VARIABLE EXPLORER
# ==========================================
# Los tensores originales (registros) no se modifican.

registros_np = a_numpy(registros)
entradas_np = a_numpy(dict(inputs))
