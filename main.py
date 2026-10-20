import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Instrumentacion import experimento_01_q, experimento_02_qkv
from Intervencion import experimento_03_intervencion_q
from Utilidades.guardado import (
    crear_carpeta_ejecucion,
    ejecutar_y_guardar,
    finalizar_ejecucion,
)
from Utilidades.lab_comun import cargar_modelo

# ==========================================
# PROMPT ÚNICO PARA TODOS LOS EXPERIMENTOS
# ==========================================

PROMPT = "Artificial intelligence is"

# ==========================================
# CARPETA DE ESTA EJECUCIÓN: Resultados/<fecha_hora>/
# ==========================================

carpeta_ejecucion = crear_carpeta_ejecucion(PROMPT)

print("Resultados en:", carpeta_ejecucion)

# ==========================================
# MODELO (se carga una sola vez)
# ==========================================

tokenizer, model = cargar_modelo()

# ==========================================
# EXPERIMENTOS
# ==========================================

print("\n" + "#" * 50)
print("# EXPERIMENTO 01")
print("#" * 50)

resultado_01 = ejecutar_y_guardar(
    carpeta_ejecucion, "experimento_01_q",
    experimento_01_q.ejecutar, PROMPT, tokenizer, model
)

print("\n" + "#" * 50)
print("# EXPERIMENTO 02")
print("#" * 50)

resultado_02 = ejecutar_y_guardar(
    carpeta_ejecucion, "experimento_02_qkv",
    experimento_02_qkv.ejecutar, PROMPT, tokenizer, model
)

print("\n" + "#" * 50)
print("# EXPERIMENTO 03")
print("#" * 50)

resultado_03 = ejecutar_y_guardar(
    carpeta_ejecucion, "experimento_03_intervencion_q",
    experimento_03_intervencion_q.ejecutar, PROMPT, tokenizer, model
)

# ==========================================
# INFORME GENERAL DE LA EJECUCIÓN
# ==========================================

informe = finalizar_ejecucion(carpeta_ejecucion)

print("\nInforme general:", informe)
