import contextlib
import io
import json
import sys
from datetime import datetime

import numpy as np
import torch

from Utilidades.lab_comun import MODEL_PATH, RAIZ
from Utilidades.tensores import a_numpy

CARPETA_RESULTADOS = RAIZ / "Resultados"

# Excel admite como máximo 16384 columnas: si un array tiene más, el CSV
# se guarda transpuesto (y el informe lo indica).
MAX_COLUMNAS_CSV = 16000


class _Tee(io.StringIO):
    """Escribe en la consola y guarda una copia del texto."""

    def __init__(self, destino):
        super().__init__()
        self.destino = destino

    def write(self, texto):
        self.destino.write(texto)
        return super().write(texto)

    def flush(self):
        self.destino.flush()


@contextlib.contextmanager
def capturar_salida():
    """Captura lo que se imprime sin dejar de mostrarlo en la consola."""

    tee = _Tee(sys.stdout)

    with contextlib.redirect_stdout(tee):
        yield tee


def _escribir_json(ruta, datos):
    ruta.write_text(
        json.dumps(datos, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )


def crear_carpeta_ejecucion(prompt):
    """Crea Resultados/<fecha_hora>/ y guarda los datos de la ejecución."""

    ahora = datetime.now()

    carpeta = CARPETA_RESULTADOS / ahora.strftime("%Y-%m-%d_%H-%M-%S")
    carpeta.mkdir(parents=True, exist_ok=False)

    _escribir_json(
        carpeta / "ejecucion.json",
        {
            "fecha": ahora.isoformat(timespec="seconds"),
            "prompt": prompt,
            "modelo": str(MODEL_PATH.name),
        }
    )

    return carpeta


# ==========================================
# RECOLECCIÓN DE LO QUE HAY EN EL DICCIONARIO
# ==========================================

def _recolectar(valor, nombre, arrays, simples):
    """Separa un valor en arrays (tensores) y valores simples."""

    if isinstance(valor, dict):
        for clave, sub in valor.items():
            _recolectar(sub, f"{nombre}__{clave}", arrays, simples)

    elif isinstance(valor, (torch.Tensor, np.ndarray)):
        arrays[nombre] = a_numpy(valor)

    elif isinstance(valor, (int, float, str, bool)) or valor is None:
        simples[nombre] = valor


def _recolectar_resultado(resultado):
    """Recorre el diccionario de resultados de un experimento.

    Las claves *_np duplican a la clave sin sufijo (si existe), así que
    se omiten. Si no existe su versión sin sufijo (por ejemplo
    entradas_np), se guardan con el nombre sin el sufijo.
    """

    arrays = {}
    simples = {}

    for clave, valor in resultado.items():

        if clave.endswith("_np"):

            if clave[:-3] in resultado:
                continue

            clave = clave[:-3]

        _recolectar(valor, clave, arrays, simples)

    return arrays, simples


# ==========================================
# ARCHIVOS DE DATOS (.npy y .csv)
# ==========================================

def _a_matriz(array):
    """Lleva un array a 2D para escribirlo como tabla. Devuelve (matriz, nota)."""

    if array.ndim == 0:
        return array.reshape(1, 1), ""

    if array.ndim == 1:
        return array.reshape(-1, 1), "una columna, un valor por fila"

    if array.ndim == 2:
        return array, ""

    filas = int(np.prod(array.shape[:-1]))

    matriz = array.reshape(filas, array.shape[-1])

    if filas == array.shape[-2]:
        return matriz, "se omitió el eje de tamaño 1 (batch)"

    return matriz, "los ejes anteriores al último se aplanaron en filas"


def _escribir_csv(ruta, array):
    """Escribe el array como CSV con todos sus valores. Devuelve la nota."""

    matriz, nota = _a_matriz(array)

    if matriz.shape[1] > MAX_COLUMNAS_CSV:
        matriz = matriz.T
        nota = (nota + "; " if nota else "") + (
            "guardado transpuesto: filas y columnas intercambiadas"
        )

    if np.issubdtype(matriz.dtype, np.floating):
        formato = "%.9g"
    else:
        matriz = matriz.astype(np.int64)
        formato = "%d"

    np.savetxt(ruta, matriz, delimiter=",", fmt=formato)

    return nota


def _estadisticas(array):
    if array.size == 0:
        return {}

    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "media": float(array.mean()),
        "desviacion": float(array.std()),
    }


# ==========================================
# INFORME EN MARKDOWN
# ==========================================

def _fmt(x):
    if isinstance(x, (float, np.floating)):
        return f"{x:.6g}"
    return str(x)


def _valor_md(v):
    if isinstance(v, str):
        return f"`{v!r}`"
    return f"`{_fmt(v)}`"


def _tabla_md(matriz):
    """Vista de una matriz como tabla Markdown (completa o recortada)."""

    filas, columnas = matriz.shape

    if filas <= 10 and columnas <= 12:
        vf, vc, recortada = filas, columnas, False
    else:
        vf, vc, recortada = min(filas, 3), min(columnas, 8), True

    lineas = [
        "| | " + " | ".join(f"c{j}" for j in range(vc)) + " |",
        "|---|" + "---|" * vc,
    ]

    for i in range(vf):
        lineas.append(
            f"| f{i} | "
            + " | ".join(_fmt(matriz[i, j]) for j in range(vc))
            + " |"
        )

    aviso = ""

    if recortada:
        aviso = (
            f"\n_Vista parcial: {vf} de {filas} filas y {vc} de "
            f"{columnas} columnas. Los valores completos están en el CSV._\n"
        )

    return "\n".join(lineas) + "\n" + aviso


def _informe_experimento(nombre, simples, meta, arrays, salida):
    L = [f"# Informe: {nombre}", ""]

    if "prompt" in simples:
        L += [f"**Prompt:** `{simples['prompt']}`", ""]

    otros = {k: v for k, v in simples.items() if k != "prompt"}

    if otros:
        L += ["## Valores", "", "| Variable | Valor |", "|---|---|"]
        for k, v in otros.items():
            L.append(f"| `{k}` | {_valor_md(v)} |")
        L.append("")

    if meta:
        L += [
            "## Tensores", "",
            "| Variable | Forma | Tipo | Mín | Máx | Media | Desv. | Archivos |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for k, m in meta.items():
            e = m["estadisticas"]
            L.append(
                f"| `{k}` | {tuple(m['forma'])} | {m['tipo']} | "
                f"{_fmt(e.get('min', ''))} | {_fmt(e.get('max', ''))} | "
                f"{_fmt(e.get('media', ''))} | {_fmt(e.get('desviacion', ''))} | "
                f"[csv]({m['csv']}) · [npy]({m['npy']}) |"
            )

        L.append("")

        L += ["## Detalle de cada tensor", ""]

        for k, m in meta.items():
            matriz, _ = _a_matriz(arrays[k])

            L += [f"### `{k}`", "", f"Forma original: `{tuple(m['forma'])}`", ""]

            if m["nota_csv"]:
                L += [f"Nota: {m['nota_csv']}.", ""]

            L += [_tabla_md(matriz)]

    if salida.strip():
        L += ["## Salida de consola", "", "```text", salida.rstrip(), "```", ""]

    return "\n".join(L)


# ==========================================
# GUARDADO DE UN EXPERIMENTO
# ==========================================

def guardar_resultado(carpeta_ejecucion, nombre, resultado, salida=""):
    """Guarda todo el resultado de un experimento en su propia subcarpeta.

    - <variable>.csv: valores completos de cada tensor (se abre en Excel).
    - <variable>.npy: lo mismo en NumPy, sin pérdida de precisión.
    - informe.md: informe legible (valores, estadísticas, vista de cada
      tensor y la salida de consola).
    - resumen.json: lo mismo en formato para programas.
    - salida.txt: lo que el experimento imprimió en consola.
    """

    carpeta = carpeta_ejecucion / nombre
    carpeta.mkdir(parents=True, exist_ok=True)

    arrays, simples = _recolectar_resultado(resultado)

    meta = {}

    for clave, array in arrays.items():

        np.save(carpeta / f"{clave}.npy", array)
        nota = _escribir_csv(carpeta / f"{clave}.csv", array)

        meta[clave] = {
            "npy": f"{clave}.npy",
            "csv": f"{clave}.csv",
            "forma": list(array.shape),
            "tipo": str(array.dtype),
            "nota_csv": nota,
            "estadisticas": _estadisticas(array),
        }

    _escribir_json(
        carpeta / "resumen.json",
        {"valores": simples, "tensores": meta}
    )

    (carpeta / "salida.txt").write_text(salida, encoding="utf-8")

    (carpeta / "informe.md").write_text(
        _informe_experimento(nombre, simples, meta, arrays, salida),
        encoding="utf-8"
    )

    return carpeta


def ejecutar_y_guardar(carpeta_ejecucion, nombre, ejecutar, *args):
    """Ejecuta un experimento, guarda su resultado y lo devuelve."""

    with capturar_salida() as salida:
        resultado = ejecutar(*args)

    guardar_resultado(
        carpeta_ejecucion, nombre, resultado, salida.getvalue()
    )

    return resultado


# ==========================================
# INFORME GENERAL DE LA EJECUCIÓN
# ==========================================

def finalizar_ejecucion(carpeta_ejecucion):
    """Crea informe.md en la carpeta de la ejecución, enlazando cada experimento."""

    datos = json.loads(
        (carpeta_ejecucion / "ejecucion.json").read_text(encoding="utf-8")
    )

    L = [
        f"# Ejecución {carpeta_ejecucion.name}", "",
        f"- **Fecha:** {datos['fecha']}",
        f"- **Modelo:** {datos['modelo']}",
        f"- **Prompt:** `{datos['prompt']}`", "",
        "## Experimentos", "",
    ]

    for sub in sorted(p for p in carpeta_ejecucion.iterdir() if p.is_dir()):

        resumen = json.loads(
            (sub / "resumen.json").read_text(encoding="utf-8")
        )

        L += [f"### [{sub.name}]({sub.name}/informe.md)", ""]

        valores = {
            k: v for k, v in resumen["valores"].items() if k != "prompt"
        }

        if valores:
            L += ["| Variable | Valor |", "|---|---|"]
            for k, v in valores.items():
                L.append(f"| `{k}` | {_valor_md(v)} |")
            L.append("")

        L += [f"Tensores guardados: {len(resumen['tensores'])}", ""]

    ruta = carpeta_ejecucion / "informe.md"

    ruta.write_text("\n".join(L), encoding="utf-8")

    return ruta
