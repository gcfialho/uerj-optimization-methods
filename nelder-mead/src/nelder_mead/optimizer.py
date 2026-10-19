"""
Implementação do Método Simplex de Nelder–Mead em Python
========================================================

• Geração de animações da evolução dos simplexes em 2D (triângulos) e 3D (tetraedros).

Execute
    python nelder_mead.py
para ver demonstrações pré‑configuradas (função de Rosenbrock em ℝ² e esfera em ℝ³).
"""

from __future__ import annotations

from typing import Callable, List, Tuple, cast, Dict, Any

import argparse
import sys
import csv
from pathlib import Path
from tabulate import tabulate

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =====================================================
# Núcleo: Nelder–Mead genérico
# =====================================================


class NelderMeadOptimizer:
    """Implementa o algoritmo Nelder–Mead para *qualquer* dimensão.

    Parameters
    ----------
    func : Callable[[np.ndarray], float]
        Função objetivo.
    initialSimplex : np.ndarray
        Array (n+1, n) com os vértices iniciais.
    alpha, gamma, rho, sigma : float
        Parâmetros de reflexão, expansão, contração e redução.
    maxIter : int
        Iterações máximas.
    tol : float
        Critério de parada (desvio‑padrão dos valores de f no simplex).
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        initialSimplex: np.ndarray,
        *,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
        maxIter: int = 200,
        tol: float = 1e-6,
    ) -> None:
        self.func = func
        self.simplex = initialSimplex.astype(float)
        self.n = self.simplex.shape[1]
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.maxIter = maxIter
        self.tol = tol
        self.history: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # Métodos internos auxiliares
    # ------------------------------------------------------------------ #

    def _centroid(self, vertices: np.ndarray) -> np.ndarray:
        """Calcula o centróide dos *n* melhores vértices."""
        return np.mean(vertices[:-1], axis=0)

    def _order_simplex(self) -> np.ndarray:
        """Ordena o simplex pelo valor da função (crescente)."""
        values = np.apply_along_axis(self.func, 1, self.simplex)
        idx = np.argsort(values)
        self.simplex = self.simplex[idx]
        return values[idx]

    # ------------------------------------------------------------------ #
    # Passo principal
    # ------------------------------------------------------------------ #

    def optimize(self) -> Tuple[np.ndarray, float]:
        """Executa o algoritmo Nelder-Mead.

        Iterativamente ajusta o simplex (conjunto de n+1 pontos em n dimensões)
        usando operações de reflexão, expansão, contração e redução
        para encontrar o mínimo da função objetivo.

        O histórico de cada simplex a cada iteração é armazenado em `self.history`.

        Returns
        -------
        Tuple[np.ndarray, float]
            Uma tupla contendo o melhor ponto encontrado (vértice do simplex final
            com o menor valor da função) e o valor da função nesse ponto.
        """
        for _ in range(self.maxIter):
            values = self._order_simplex()
            self.history.append(self.simplex.copy())

            if np.std(values) < self.tol:
                break

            centroid = self._centroid(self.simplex)
            worst = self.simplex[-1]

            # Reflexão --------------------------------------------------
            reflected = centroid + self.alpha * (centroid - worst)
            fReflected = self.func(reflected)

            if values[0] <= fReflected < values[-2]:
                self.simplex[-1] = reflected
                continue

            # Expansão --------------------------------------------------
            if fReflected < values[0]:
                expanded = centroid + self.gamma * (reflected - centroid)
                fExpanded = self.func(expanded)
                self.simplex[-1] = expanded if fExpanded < fReflected else reflected
                continue

            # Contração -----------------------------------------------
            contracted = centroid + self.rho * (worst - centroid)
            fContracted = self.func(contracted)
            if fContracted < values[-1]:
                self.simplex[-1] = contracted
                continue

            # Redução --------------------------------------------------
            best = self.simplex[0]
            self.simplex = best + self.sigma * (self.simplex - best)

        # Guardar estado final
        self._order_simplex()
        self.history.append(self.simplex.copy())
        bestPoint = self.simplex[0]
        bestValue = self.func(bestPoint)
        return bestPoint, bestValue


# =====================================================
# Visualização 2D: triângulos no plano
# =====================================================


class SimplexVisualizer2D:
    """Anima triângulos sobre curvas de nível."""

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        history: List[np.ndarray],
        *,
        levels: int = 50,
        title: str | None = None,
    ) -> None:
        self.func = func
        self.bounds = bounds
        self.history = history
        self.levels = levels
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax: Axes
        if title:
            self.fig.suptitle(title, fontsize=14)
        self._prepare_background()

    def _prepare_background(self) -> None:
        (xMin, xMax), (yMin, yMax) = self.bounds
        x = np.linspace(xMin, xMax, 400)
        y = np.linspace(yMin, yMax, 400)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self.func(np.array([xx, yy])) for xx in x] for yy in y])
        self.ax.contourf(X, Y, Z, self.levels, alpha=0.6, cmap="viridis")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

    def _update(self, i: int, poly):
        verts = self.history[i]
        poly.set_xy(verts)
        poly.set_edgecolor("white" if i == len(self.history) - 1 else "red")
        return poly,

    def animate(self, *, savePath: str | None = None) -> None:
        if not self.history:
            print("[!] Animação 2D pulada: Histórico vazio.")
            plt.close(self.fig)
            return
        
        poly = Polygon(self.history[0], fill=False, color="red", linewidth=2)
        self.ax.add_patch(poly)
        anim = animation.FuncAnimation(
            self.fig,
            self._update,
            fargs=(poly,),
            frames=len(self.history),
            interval=300,
            blit=True,
        )
        if savePath:
            anim.save(savePath, writer="ffmpeg", fps=3)
        else:
            plt.show()


# =====================================================
# Visualização 3D: tetraedros no espaço
# =====================================================


class SimplexVisualizer3D:
    """Anima a evolução de tetraedros (simplexes) no espaço 3-D."""

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        history: List[np.ndarray],
        *,
        showSurface: bool = True,
        title: str | None = None,
    ) -> None:
        self.func = func
        self.bounds = bounds
        self.history = history
        self.showSurface = showSurface

        # ---------- figura e eixo ----------
        self.fig = plt.figure(figsize=(8, 6))
        if title:
            self.fig.suptitle(title, fontsize=14)
        # Declara o tipo primeiro
        self.ax: Axes3D
        # Depois atribui, usando cast para o linter
        self.ax = cast(Axes3D, self.fig.add_subplot(111, projection="3d"))
        self._setup_axes()
        if self.showSurface:
            self._plot_surface()

        # ---------- cria UM Poly3DCollection (será atualizado) ----------
        if not self.history:
            print("[!] Poly3DCollection não criado: Histórico vazio.")
            self.poly = None
        else:
            verts = self._faces(self.history[0])
            # cor rubi bem transparente → praticamente invisível, porém válida
            self.poly = Poly3DCollection(
                verts, facecolor=(0.8, 0.1, 0.1, 0.25), edgecolor="red", linewidths=1
            )
            self.ax.add_collection3d(self.poly)

    # ------------------------------------------------------------------ #
    # utilitários
    # ------------------------------------------------------------------ #
    def _setup_axes(self) -> None:
        (xMin, xMax), (yMin, yMax), (zMin, zMax) = self.bounds
        self.ax.set_xlim(xMin, xMax)
        self.ax.set_ylim(yMin, yMax)
        self.ax.set_zlim(zMin, zMax)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    def _plot_surface(self, res: int = 40) -> None:
        """Desenha uma superfície-guia da função no plano z = 0."""
        (xMin, xMax), (yMin, yMax), _ = self.bounds
        xs = np.linspace(xMin, xMax, res)
        ys = np.linspace(yMin, yMax, res)
        xsM, ysM = np.meshgrid(xs, ys)
        zsM = np.vectorize(
            lambda x, y: self.func(np.array([x, y, 0.0]))
        )(xsM, ysM)

        self.ax.plot_surface(
            xsM, ysM, zsM,
            cmap="viridis", alpha=0.20, linewidth=0, antialiased=False
        )

    @staticmethod
    def _faces(tetra: np.ndarray):
        """Retorna as quatro faces triangulares de um tetraedro."""
        return [
            [tetra[0], tetra[1], tetra[2]],
            [tetra[0], tetra[1], tetra[3]],
            [tetra[0], tetra[2], tetra[3]],
            [tetra[1], tetra[2], tetra[3]],
        ]

    # ------------------------------------------------------------------ #
    # animação
    # ------------------------------------------------------------------ #
    def _update(self, frame: int):
        """Atualiza os vértices do poliedro 3D para o quadro da animação.

        Este método é chamado pela `FuncAnimation` para cada `frame`.
        Ele busca o estado do simplex correspondente ao `frame` no histórico
        e atualiza a geometria do objeto `Poly3DCollection`.
        """
        if self.poly is None:
            return ()
            
        tetra = self.history[frame]
        self.poly.set_verts(self._faces(tetra))

        # realça o simplex final
        if frame == len(self.history) - 1:
            self.poly.set_edgecolor("red")
            self.poly.set_facecolor((1, 1, 1, 0.8))
            self.poly.set_alpha(0.9)
        return (self.poly,)

    def animate(
        self, *, interval: int = 300, savePath: str | None = None
    ) -> None:
        """
        Exibe (ou grava) a animação 3-D.
        - interval: milissegundos entre quadros.
        - savePath: se fornecido, grava .mp4 via ffmpeg; caso contrário, mostra.
        """
        if self.poly is None or not self.history:
            print("[!] Animação 3D pulada: Histórico vazio ou Poly3DCollection não criado.")
            plt.close(self.fig)
            return

        ani = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=len(self.history),
            interval=interval,
            blit=False,        # ← blit=False é frequentemente necessário para evitar
                               #   artefatos visuais em animações 3D complexas.
            repeat=False,
        )

        if savePath:
            ani.save(savePath, writer="ffmpeg", fps=3)
        else:
            plt.show()

# =====================================================
# Utilitário para Executar e Visualizar um Problema
# =====================================================

def run_problem(problem: Dict[str, Any], save: bool, show: bool) -> None:
    """Executa um único problema de otimização e lida com os outputs."""
    name = problem["name"]
    func = problem["func"]
    simplex0 = problem["simplex0"]
    dim = problem["dim"]
    bounds = problem["bounds"]
    opt_params = problem.get("optimizer_params", {})
    viz_params = problem.get("viz_params", {})

    print(f"\n[-- Executando Problema: {name} ({dim}D) --]")
    print(f"  Função: {func.__name__}")
    print(f"  Simplex inicial:\n{simplex0}".replace("\n", "\n  "))

    # Instancia e executa o otimizador
    optimizer = NelderMeadOptimizer(func, simplex0, **opt_params)
    print("\n  Otimizando...")
    best_point, best_value = optimizer.optimize()
    print(f"    -> Ponto ótimo encontrado: {best_point}")
    print(f"    -> Valor da função: {best_value:.4e}")
    iterations = len(optimizer.history) -1 if optimizer.history else 0
    print(f"    -> Total de iterações: {iterations}")

    # --- Processar histórico para tabela e CSV ---
    if optimizer.history:
        print("\n  Processando histórico de otimização...")
        table_data = []
        # Define cabeçalhos dinamicamente com base na dimensão
        coord_names = [f"x{i+1}" for i in range(dim)]
        headers = ["Iteração", *coord_names, f"f({','.join(coord_names)})"]

        for i, simplex in enumerate(optimizer.history):
            best_point_iter = simplex[0]
            value_iter = func(best_point_iter)
            row = [i, *best_point_iter, value_iter]
            table_data.append(row)

        # Imprimir tabela no console
        print("\n  Histórico dos Melhores Pontos por Iteração:")
        print("  " + tabulate(table_data, headers=headers, floatfmt=".6e", tablefmt="grid").replace("\n", "\n  "))

        # Salvar CSV se solicitado
        if save:
            output_dir = Path(f"{dim}d")
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / f"{name}_history.csv"
            try:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(table_data)
                print(f"\n  [+] Histórico salvo em CSV: {csv_path.resolve()}")
            except IOError as e:
                print(f"\n  [!] Erro ao salvar CSV para {name}: {e}")
    else:
        print("\n  Nenhum histórico de otimização para processar.")

    # --- Visualização --- 
    if show or save:
        if not optimizer.history:
            print("\n  Visualização pulada (sem histórico).")
            return
        
        print(f"\n  Preparando visualização {dim}D...")
        visualizer: SimplexVisualizer2D | SimplexVisualizer3D | None = None
        save_file_path: Path | None = None

        output_dir = Path(f"{dim}d")
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_file_path = output_dir / f"{name}.mp4"

        try:
            # Passa o nome do problema como título para o visualizador
            common_viz_args = {
                "func": func, 
                "bounds": bounds, 
                "history": optimizer.history,
                "title": name
            }
            
            if dim == 2:
                visualizer = SimplexVisualizer2D(**common_viz_args, **viz_params)
            elif dim == 3:
                visualizer = SimplexVisualizer3D(**common_viz_args, **viz_params)
            else:
                print(f"  [!] Visualização não suportada para dimensão {dim}")

            if visualizer:
                if save_file_path:
                    print(f"    Salvando animação {dim}D em: {save_file_path.resolve()}")
                    visualizer.animate(savePath=str(save_file_path))
                    print(f"      -> [{dim}D] Animação salva.")
                
                if show:
                    print(f"    Exibindo animação {dim}D interativa...")
                    visualizer.animate()
                    print(f"      -> Janela de animação {dim}D fechada.")

        except Exception as e:
            print(f"\n  [!] Erro durante a visualização para {name}: {e}")

    else:
        print(f"\n  Visualização {dim}D desativada.")

    print(f"\n[-- Fim da Execução: {name} --]")

# =====================================================
# Seção Main (apenas para teste rápido do módulo)
# =====================================================


def _run_demo_2d_internal() -> None:
    # Função interna para evitar poluir namespace se importado
    print("Demo 2D: Rosenbrock (interno ao módulo)")
    # Redefinir rosenbrock localmente para a demo
    def rosenbrock_local(point: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
        x, y = point
        return (a - x)**2 + b * (y - x**2)**2
    
    initialSimplex = np.array([[-1.2, 1.0], [0.0, 1.2], [-1.0, 0.8]])
    optimizer = NelderMeadOptimizer(rosenbrock_local, initialSimplex, maxIter=250)
    bestPoint, bestValue = optimizer.optimize()
    print(f"Ponto ótimo: {bestPoint}, valor: {bestValue:.4e}")
    viz = SimplexVisualizer2D(rosenbrock_local, [(-2.0, 2.0), (-1.0, 3.0)], optimizer.history)
    viz.animate()


def _run_demo_3d_internal() -> None:
    # Função interna
    print("\nDemo 3D: Esfera (interno ao módulo)")
    # Redefinir sphere3d localmente para a demo
    def sphere3d_local(point: np.ndarray) -> float:
        return np.sum(np.square(point))

    initialSimplex = np.array(
        [[2.0, 1.0, -1.0], [-1.5, 2.0, 0.5], [1.0, -2.0, 1.5], [0.0, 0.5, 2.0]]
    )
    optimizer = NelderMeadOptimizer(sphere3d_local, initialSimplex, maxIter=300)
    bestPoint, bestValue = optimizer.optimize()
    print(f"Ponto ótimo: {bestPoint}, valor: {bestValue:.4e}")
    viz = SimplexVisualizer3D(
        sphere3d_local,
        [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
        optimizer.history,
    )
    viz.animate()


if __name__ == "__main__":
    # Mantém a capacidade de rodar demos simples diretamente do módulo
    parser = argparse.ArgumentParser(description="Nelder–Mead Demos Internas do Módulo")
    parser.add_argument(
        "demo",
        choices=["2d", "3d", "all"],
        default="all",
        nargs="?",
        help="Escolha qual demonstração interna rodar (2d, 3d ou all)",
    )
    args = parser.parse_args()

    try:
        if args.demo in ["2d", "all"]:
            _run_demo_2d_internal()
        if args.demo in ["3d", "all"]:
            _run_demo_3d_internal()
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.", file=sys.stderr)
