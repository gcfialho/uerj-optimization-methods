"""main.py
============

Script de demonstração que utiliza o módulo ``nelder_mead`` para resolver
uma lista pré-definida de problemas de otimização, exibindo e/ou salvando
os resultados (histórico e animação).

Uso
---
$ python main.py            # Executa todos os problemas, mostra janelas interativas e tabelas
$ python main.py --save     # Salva animações MP4 e históricos CSV nos diretórios 2d/ ou 3d/
$ python main.py --noshow   # Executa sem exibir janelas Matplotlib (útil com --save)
"""
from __future__ import annotations

import argparse
import numpy as np
import sys
from typing import List, Dict, Any, Callable

# ---------------------------------------------------------------------------
# Importa as classes e funções do módulo nelder_mead
# ---------------------------------------------------------------------------
try:
    from nelder_mead import (
        NelderMeadOptimizer,
        SimplexVisualizer2D,
        SimplexVisualizer3D,
        run_problem,
    )
except ImportError as exc:
    sys.exit(
        "Não foi possível importar 'nelder_mead'. Verifique se o arquivo está "
        "no mesmo diretório ou se o pacote foi instalado no PYTHONPATH.\n"
        f"Detalhes: {exc}"
    )

# ---------------------------------------------------------------------------
# Funções-objetivo de teste
# ---------------------------------------------------------------------------

def booth(point: np.ndarray) -> float:
    """Função de Booth (mínimo global em (1, 3) com f=0)."""
    x, y = point
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

def sphere3d(point: np.ndarray) -> float:
    """Função esfera (mínimo global na origem)."""
    return np.sum(np.square(point))

def rosenbrock(point: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    """Clássica função de Rosenbrock em 2D."""
    x, y = point
    return (a - x)**2 + b * (y - x**2)**2

def shifted_sphere3d(point: np.ndarray) -> float:
    """Função esfera deslocada (mínimo global em (1, -2, 0.5) com f=0)."""
    offset = np.array([1.0, -2.0, 0.5])
    return np.sum(np.square(point - offset))

def himmelblau(point: np.ndarray) -> float:
    """Função de Himmelblau (4 mínimos em (3,2), (-2.8,3.1), (-3.7,-3.2), (3.5,-1.8)). f=0"""
    x, y = point
    term1 = (x**2 + y - 11)**2
    term2 = (x + y**2 - 7)**2
    return term1 + term2

def paraboloid3d(point: np.ndarray) -> float:
    """Paraboloide simples: f(x,y,z) = x² + y² + z² (igual à sphere3d, mas com outro nome/simplex)"""
    # É a mesma função sphere3d, mas podemos usar um simplex inicial diferente
    return np.sum(np.square(point))


# ---------------------------------------------------------------------------
# Definição dos Problemas a serem Resolvidos
# ---------------------------------------------------------------------------

# Lista de problemas. Cada item é um dicionário descrevendo uma execução.
PROBLEMS: List[Dict[str, Any]] = [
    {
        "name": "Booth_2D",
        "func": booth,
        "simplex0": np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]),
        "dim": 2,
        "bounds": [(-10.0, 10.0), (-10.0, 10.0)],
        "optimizer_params": {"maxIter": 100, "tol": 1e-7},
        "viz_params": {"levels": 50}
    },
    {
        "name": "Sphere_3D",
        "func": sphere3d,
        "simplex0": np.array([
            [5.0, -3.0, 2.0],
            [5.05, -3.0, 2.0],
            [5.0, -2.95, 2.0],
            [5.0, -3.0, 2.05],
        ]),
        "dim": 3,
        "bounds": [(-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)],
        "optimizer_params": {"maxIter": 150},
        "viz_params": {"showSurface": True}
    },
    {
        "name": "Rosenbrock_2D",
        "func": rosenbrock,
        "simplex0": np.array([[-1.2, 1.0], [0.0, 1.2], [-1.0, 0.8]]),
        "dim": 2,
        "bounds": [(-2.0, 2.0), (-1.0, 3.0)],
        "optimizer_params": {"maxIter": 250, "tol": 1e-6},
        "viz_params": {"levels": 60}
    },
    {
        "name": "Shifted_Sphere_3D",
        "func": shifted_sphere3d,
        "simplex0": np.array([
            [-1.0, -1.0, -1.0],
            [-0.9, -1.0, -1.0],
            [-1.0, -0.9, -1.0],
            [-1.0, -1.0, -0.9],
        ]),
        "dim": 3,
        "bounds": [(-3.0, 3.0), (-4.0, 0.0), (-2.0, 2.0)],
        "optimizer_params": {},
        "viz_params": {"showSurface": False}
    },
    {
        "name": "Himmelblau_2D",
        "func": himmelblau,
        "simplex0": np.array([[-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]),
        "dim": 2,
        "bounds": [(-6.0, 6.0), (-6.0, 6.0)],
        "optimizer_params": {"maxIter": 150, "tol": 1e-7},
        "viz_params": {"levels": 100}
    },
    {
        "name": "Paraboloid_3D",
        "func": paraboloid3d,
        "simplex0": np.array([
            [-4.0, -4.0, -4.0],
            [-3.9, -4.0, -4.0],
            [-4.0, -3.9, -4.0],
            [-4.0, -4.0, -3.9],
        ]),
        "dim": 3,
        "bounds": [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
        "optimizer_params": {},
        "viz_params": {"showSurface": True}
    },
]

# ---------------------------------------------------------------------------
# Ponto de Entrada Principal
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa demonstrações do otimizador Nelder-Mead a partir de uma lista de problemas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Salva animações em .mp4 e históricos em .csv nos diretórios 2d/ ou 3d/."
    )
    parser.add_argument(
        "--noshow", action="store_true",
        help="Não exibe janelas interativas do Matplotlib (útil com --save)."
    )

    args = parser.parse_args()
    save_outputs = args.save
    show_plots = not args.noshow

    if not save_outputs and not show_plots:
        print("[AVISO] Nenhuma ação de output (--save ou não usar --noshow) especificada.")
        print("         Apenas as tabelas de histórico serão impressas no console.")

    print("\nIniciando demonstrações Nelder-Mead...")
    print("=" * 70)

    num_problems = len(PROBLEMS)
    for i, problem_config in enumerate(PROBLEMS):
        print(f"Processando problema {i+1}/{num_problems}: {problem_config.get('name', 'NOME_NAO_DEFINIDO')}")
        run_problem(problem_config, save=save_outputs, show=show_plots)
        if i < num_problems - 1:
             print("-" * 70)

    print("=" * 70)
    print("Demonstrações concluídas.")


if __name__ == "__main__":
    main()
