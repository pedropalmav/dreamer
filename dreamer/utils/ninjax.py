import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_leaves, tree_flatten, tree_unflatten
from typing import Callable, Any, Optional, Tuple


def scan(
    fun: Callable,
    carry: Any,
    xs: Any,
    length: Optional[int] = None,
    reverse: bool = False,
    axis: int = 0,
) -> Tuple[Any, Any]:
    """
    Aplica una función iterativamente sobre una secuencia.

    Args:
        fun: Función (carry, x) -> (new_carry, y)
        carry: Estado inicial que se pasa entre iteraciones
        xs: Secuencia de inputs (tensor o pytree de tensors)
        length: Longitud explícita de la secuencia
        reverse: Si True, itera en orden inverso
        axis: Dimensión sobre la cual iterar

    Returns:
        (final_carry, ys): Carry final y secuencia de outputs
    """

    # 1. Mover el eje de iteración al principio si es necesario
    if axis != 0:
        xs = tree_map(
            lambda x: x.transpose(0, axis) if isinstance(x, torch.Tensor) else x, xs
        )

    # 2. Determinar longitud
    if length is None:
        leaves = tree_leaves(xs)
        length = len(leaves[0]) if leaves else 0

    # 3. Extraer secuencia de tensors individuales
    def get_sequence(tree, idx):
        """Extrae el elemento idx de cada tensor en el pytree"""
        return tree_map(lambda x: x[idx] if isinstance(x, torch.Tensor) else x, tree)

    # 4. Aplicar scan
    indices = range(length - 1, -1, -1) if reverse else range(length)

    ys_list = []
    current_carry = carry

    for i in indices:
        x_i = get_sequence(xs, i)
        current_carry, y_i = fun(current_carry, x_i)
        ys_list.append(y_i)

    # 5. Revertir si fue reverse
    if reverse:
        ys_list = ys_list[::-1]

    # 6. Apilar outputs en tensors
    # Primero flatten para obtener estructura
    flat_first, spec = tree_flatten(ys_list[0])

    # Apilar cada posición en flat
    stacked_flat = []
    for i in range(len(flat_first)):
        leaf_sequence = [tree_flatten(y)[0][i] for y in ys_list]
        if isinstance(leaf_sequence[0], torch.Tensor):
            stacked_flat.append(torch.stack(leaf_sequence, dim=0))
        else:
            stacked_flat.append(leaf_sequence)

    # Reconstruir estructura
    ys = tree_unflatten(stacked_flat, spec)

    # 7. Restaurar el eje original
    if axis != 0:
        ys = tree_map(
            lambda y: y.transpose(0, axis) if isinstance(y, torch.Tensor) else y, ys
        )

    return current_carry, ys


# Versión simplificada (más común en la práctica)
def simple_scan(
    fun: Callable, carry: Any, xs: Any, reverse: bool = False
) -> Tuple[Any, Any]:
    """
    Versión simplificada de scan para PyTorch.
    Asume que xs tiene la dimensión de tiempo en axis=0.
    """
    # Determinar longitud
    leaves = tree_leaves(xs)
    length = len(leaves[0]) if leaves else 0

    # Aplicar scan
    indices = range(length - 1, -1, -1) if reverse else range(length)

    ys_list = []
    current_carry = carry

    for i in indices:
        # Extraer elemento i de cada tensor en el pytree
        x_i = tree_map(lambda x: x[i] if isinstance(x, torch.Tensor) else x, xs)
        current_carry, y_i = fun(current_carry, x_i)
        ys_list.append(y_i)

    # Revertir si fue reverse
    if reverse:
        ys_list = ys_list[::-1]

    # Apilar outputs
    flat_first, spec = tree_flatten(ys_list[0])
    stacked_flat = []
    for i in range(len(flat_first)):
        leaf_sequence = [tree_flatten(y)[0][i] for y in ys_list]
        if isinstance(leaf_sequence[0], torch.Tensor):
            stacked_flat.append(torch.stack(leaf_sequence, dim=0))
        else:
            stacked_flat.append(leaf_sequence)

    ys = tree_unflatten(stacked_flat, spec)

    return current_carry, ys
