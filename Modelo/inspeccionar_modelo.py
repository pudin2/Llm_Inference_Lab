import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

MODEL_PATH = (
    r"D:\Proyectos\Proyectos Python"
    r"\Llm-Lab\Modelo\SmolLM2-135M"
)

torch.set_num_threads(2)

print("=== LABORATORIO LLM ===")

# ==========================================
# 2. CARGAR TOKENIZADOR
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Tokenizador cargado.")

# ==========================================
# 3. CARGAR MODELO
# ==========================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    attn_implementation="eager",
    local_files_only=True
)

model.to("cpu")
model.eval()

print("Modelo cargado.")

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

texto = "Artificial intelligence is"

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