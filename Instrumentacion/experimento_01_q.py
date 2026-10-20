import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from Utilidades.lab_comun import cargar_modelo, leer_prompt
from Utilidades.prediccion import siguiente_token
from Utilidades.tensores import a_numpy


def ejecutar(prompt, tokenizer, model):
    """Experimento 01: instrumentación de la proyección Q de la capa 0.

    Devuelve un diccionario con los tensores capturados, sus copias en
    NumPy (sufijo _np) y el token que predice el modelo.
    """

    # 1. Seleccionar la primera proyección Q
    q_proj = model.model.layers[0].self_attn.q_proj

    print("=== MATRIZ Wq ===")
    print("Dimensiones:", tuple(q_proj.weight.shape))

    # 2. Registrar datos de la operación
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

    # 3. Instalar hook
    handle = q_proj.register_forward_hook(hook_q)

    # 4. Ejecutar inferencia
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    try:
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False)

    finally:
        # 5. Retirar hook
        handle.remove()

    # 6. Mostrar resultados
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

    # 7. Validar la operación
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

    # 8. Token que predice el modelo
    token_id, token_predicho = siguiente_token(tokenizer, outputs.logits)

    # 9. Resultados: tensores originales y copias en NumPy para el
    # Variable Explorer de Spyder.
    return {
        "prompt": prompt,
        "token_id": token_id,
        "token_predicho": token_predicho,
        "registro": registro,
        "Q_recalculado": Q_recalculado,
        "registro_np": a_numpy(registro),
        "Q_recalculado_np": a_numpy(Q_recalculado),
        "entradas_np": a_numpy(dict(inputs)),
    }


if __name__ == "__main__":
    texto = leer_prompt("Experimento 01: instrumentación de Q")

    tokenizer, model = cargar_modelo()

    resultado = ejecutar(texto, tokenizer, model)
