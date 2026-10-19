import torch


def a_numpy(obj):
    """Devuelve una copia en NumPy de un tensor (o de un dict de tensores).

    El original no se modifica. Spyder solo muestra los valores de una
    variable en su Variable Explorer si es un array de NumPy.
    """

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().copy()

    if isinstance(obj, dict):
        return {k: a_numpy(v) for k, v in obj.items()}

    return obj
