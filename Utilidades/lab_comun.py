import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RAIZ = Path(__file__).resolve().parents[1]

MODEL_PATH = RAIZ / "Modelo" / "SmolLM2-135M"

PROMPT_DEFECTO = "Artificial intelligence is"


def leer_prompt(descripcion=None):
    """Devuelve el prompt indicado con --prompt (o el valor por defecto)."""

    parser = argparse.ArgumentParser(description=descripcion)

    parser.add_argument(
        "--prompt",
        default=None,
        help="Texto de entrada. Si se omite, se pregunta por consola."
    )

    # parse_known_args tolera los argumentos extra que añaden los IDE.
    prompt = parser.parse_known_args()[0].prompt

    if prompt is not None:
        return prompt

    # Modo interactivo (consola de Spyder / VS Code / terminal).
    try:
        escrito = input(f"Prompt [{PROMPT_DEFECTO}]: ").strip()
    except EOFError:
        escrito = ""

    return escrito or PROMPT_DEFECTO


def cargar_modelo(num_threads=2):
    """Carga tokenizador y modelo en CPU, en modo evaluación."""

    torch.set_num_threads(num_threads)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True
    )

    model.to("cpu")
    model.eval()

    return tokenizer, model
