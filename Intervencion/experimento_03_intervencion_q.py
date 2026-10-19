import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from Utilidades.lab_comun import cargar_modelo, leer_prompt
from Utilidades.prediccion import siguiente_token
from Utilidades.tensores import a_numpy

# ==========================================
# 1. CARGAR MODELO
# ==========================================

texto = leer_prompt("Experimento 03: intervención de Q")

tokenizer, model = cargar_modelo()

# ==========================================
# 2. PREPARAR ENTRADA
# ==========================================

inputs = tokenizer(
    texto,
    return_tensors="pt"
)

capturas = {}

def capturar_q_original(module, inputs, output):
    capturas["Q_original"] = output.detach().clone()

hook_original = (
    model.model.layers[0].self_attn.q_proj
    .register_forward_hook(capturar_q_original)
)

# ==========================================
# 3. INFERENCIA ORIGINAL
# ==========================================

with torch.inference_mode():

    original = model(
        **inputs,
        use_cache=False
    ).logits.clone()

print("Inferencia original completada.")

hook_original.remove()

# ==========================================
# 4. IMPLEMENTACIÓN EXPERIMENTAL
# ==========================================

class LinearIntervenida(nn.Module):

    def __init__(self, original):

        super().__init__()

        # Conservamos el módulo y sus pesos.
        self.original = original

        self.multiplicaciones = 0
        self.sumas = 0

    def forward(self, X):

        W = self.original.weight
        bias = self.original.bias

        M = X.numel() // W.shape[1]
        N, K = W.shape

        # Multiplicación elemento a elemento.
        # Resultado: [batch, tokens, N, K]

        productos = (
            X.unsqueeze(-2) * W
        )

        # Suma explícita sobre K.
        Q = productos.sum(dim=-1)

        if bias is not None:
            Q = Q + bias

        # Conteo algorítmico.
        self.multiplicaciones += M * N * K

        self.sumas += M * N * (K - 1)

        if bias is not None:
            self.sumas += M * N

        return Q

# ==========================================
# 5. SUSTITUIR Q
# ==========================================

attention = model.model.layers[0].self_attn

q_original = attention.q_proj

q_intervenida = LinearIntervenida(q_original)

attention.q_proj = q_intervenida

def capturar_q_intervenida(module, inputs, output):
    capturas["Q_intervenida"] = output.detach().clone()

hook_intervenido = q_intervenida.register_forward_hook(
    capturar_q_intervenida
)

print("Proyección Q sustituida.")

# ==========================================
# 6. INFERENCIA INTERVENIDA
# ==========================================

try:

    with torch.inference_mode():

        intervenido = model(
            **inputs,
            use_cache=False
        ).logits.clone()

finally:

    # Restaurar el módulo original.
    attention.q_proj = q_original

print("Inferencia intervenida completada.")
print("Modelo restaurado.")

# ==========================================
# 7. COMPARACIÓN
# ==========================================

diferencia = (
    original - intervenido
).abs()

print("\n=== OPERACIONES ===")

print(
    "Multiplicaciones:",
    q_intervenida.multiplicaciones
)

print(
    "Sumas:",
    q_intervenida.sumas
)

print("\n=== VALIDACIÓN ===")

print(
    "Logits equivalentes:",
    torch.allclose(
        original,
        intervenido,
        rtol=1e-5,
        atol=1e-6
    )
)

print(
    "Diferencia máxima:",
    diferencia.max().item()
)

print(
    "Diferencia media:",
    diferencia.mean().item()
)

token_original = original[0, -1].argmax().item()

token_intervenido = intervenido[0, -1].argmax().item()

print("\nToken original:", token_original)

print("Token intervenido:", token_intervenido)

print(
    "Mismo token:",
    token_original == token_intervenido
)

print("\nEXPERIMENTO FINALIZADO")

# ==========================================
# ANÁLISIS DE PRECISIÓN NUMÉRICA
# ==========================================

print("\n=== COMPARACIÓN DIRECTA DE Q ===")

Q_original = capturas["Q_original"]
Q_intervenida = capturas["Q_intervenida"]

diff_q = (Q_original - Q_intervenida).abs()

print("Diferencia máxima Q:", diff_q.max().item())
print("Diferencia media Q:", diff_q.mean().item())

print(
    "Q equivalente (1e-5):",
    torch.allclose(
        Q_original,
        Q_intervenida,
        rtol=1e-5,
        atol=1e-5
    )
)

print("\n=== COMPARACIÓN DE LOGITS ===")

diff_logits = (original - intervenido).abs()

for tolerancia in [1e-6, 1e-5, 1e-4]:

    equivalentes = torch.allclose(
        original,
        intervenido,
        rtol=tolerancia,
        atol=tolerancia
    )

    print(
        f"Tolerancia {tolerancia}:",
        equivalentes
    )

print("\n=== ELEMENTOS FUERA DE TOLERANCIA ===")

limite = 1e-5 + 1e-5 * intervenido.abs()

fuera = (diff_logits > limite).sum().item()

print("Logits fuera de tolerancia:", fuera)
print("Total de logits:", diff_logits.numel())

print("\n=== DECISIÓN DEL SIGUIENTE TOKEN ===")

valores, indices = torch.topk(
    original[0, -1],
    k=2
)

margen = (valores[0] - valores[1]).item()

print("Primer token:", indices[0].item())
print("Segundo token:", indices[1].item())
print("Margen entre ambos:", margen)

print(
    "Diferencia máxima última posición:",
    diff_logits[0, -1].max().item()
)

_, token_original_texto = siguiente_token(tokenizer, original)
_, token_intervenido_texto = siguiente_token(tokenizer, intervenido)

# ==========================================
# COPIAS EN NUMPY PARA EL VARIABLE EXPLORER
# ==========================================
# Los tensores originales (capturas, original, intervenido) no se modifican.

capturas_np = a_numpy(capturas)
logits_original_np = a_numpy(original)
logits_intervenido_np = a_numpy(intervenido)
