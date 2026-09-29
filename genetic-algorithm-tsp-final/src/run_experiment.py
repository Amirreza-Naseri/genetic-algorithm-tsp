from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from genetic_tsp import (
    GAConfig,
    GeneticTSPSolver,
    euclidean_distance_matrix,
    greedy_nearest_neighbor,
    tour_length,
)


def make_demo_coords(n_cities: int, seed: int, scale: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n_cities, 2)) * scale


def load_coords(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    names = set(data.dtype.names or ())
    if not {"x", "y"}.issubset(names):
        raise ValueError("Coordinate CSV must have headers: city,x,y (x and y are required).")
    return np.column_stack([data["x"], data["y"]]).astype(float)


def save_coords(coords: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["city", "x", "y"])
        for i, (x, y) in enumerate(coords, start=1):
            w.writerow([i, f"{x:.8f}", f"{y:.8f}"])


def plot_route(coords: np.ndarray, tour: list[int], best_len: float, out: Path) -> None:
    idx = np.asarray(tour, dtype=int) - 1
    closed = np.append(idx, idx[0])
    plt.figure(figsize=(8, 7))
    plt.plot(coords[closed, 0], coords[closed, 1], marker="o")
    for city, (x, y) in enumerate(coords, start=1):
        plt.annotate(str(city), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.title(f"Best GA tour — length {best_len:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()


def plot_convergence(history: list[float], out: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(history)), history)
    plt.title("Genetic Algorithm Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best tour length")
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()


def plot_baselines(random_mean: float, greedy_len: float, ga_len: float, out: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.bar(["Random mean", "Greedy", "Genetic Algorithm"], [random_mean, greedy_len, ga_len])
    plt.ylabel("Tour length (lower is better)")
    plt.title("Baseline Comparison")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    images_dir = root / "images"
    images_dir.mkdir(exist_ok=True)

    if args.coords:
        coords = load_coords(Path(args.coords))
    else:
        coords = make_demo_coords(args.n_cities, args.seed)
        save_coords(coords, Path("data") / "demo_cities.csv")

    dist = euclidean_distance_matrix(coords)
    n = len(coords)

    # Reproducible random-tour baseline.
    baseline_rng = np.random.default_rng(args.seed)
    random_lengths = []
    for _ in range(args.random_baseline_samples):
        tour = baseline_rng.permutation(np.arange(1, n + 1)).tolist()
        random_lengths.append(tour_length(tour, dist))
    random_mean = float(np.mean(random_lengths))

    greedy_tour = greedy_nearest_neighbor(dist, start=0)
    greedy_len = tour_length(greedy_tour, dist)

    cfg = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        tournament_k=args.tournament_k,
        elite_size=args.elite_size,
        stall_generations=args.stall_generations,
        seed_greedy_fraction=args.seed_greedy_fraction,
        random_seed=args.seed,
    )
    solver = GeneticTSPSolver(dist, cfg)
    best_tour, best_len, history = solver.solve()

    improvement = 100.0 * (greedy_len - best_len) / greedy_len

    with (root / "best_tour.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["visit_order", "city"])
        for order, city in enumerate(best_tour, start=1):
            w.writerow([order, city])

    with (root / "convergence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_length"])
        for generation, value in enumerate(history):
            w.writerow([generation, f"{value:.10f}"])

    summary = {
        "n_cities": n,
        "seed": args.seed,
        "random_tour_mean": random_mean,
        "greedy_length": greedy_len,
        "ga_best_length": best_len,
        "ga_improvement_vs_greedy_percent": improvement,
        "generations_executed": len(history) - 1,
    }
    with (root / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key, value in summary.items():
            w.writerow([key, value])

    plot_route(coords, best_tour, best_len, images_dir / "best_route.png")
    plot_convergence(history, images_dir / "convergence.png")
    plot_baselines(random_mean, greedy_len, best_len, images_dir / "baseline_comparison.png")

    print(f"Cities: {n}")
    print(f"Random-tour mean: {random_mean:.3f}")
    print(f"Greedy length: {greedy_len:.3f}")
    print(f"GA best length: {best_len:.3f}")
    print(f"Improvement vs greedy: {improvement:.2f}%")
    print(f"Generations executed: {len(history) - 1}")
    print("Best tour:", best_tour)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a reproducible GA experiment for Euclidean TSP.")
    p.add_argument("--coords", type=str, default=None, help="Optional CSV with headers city,x,y.")
    p.add_argument("--output-dir", default="results", help="Directory for CSVs and plots.")
    p.add_argument("--n-cities", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--population-size", type=int, default=120)
    p.add_argument("--generations", type=int, default=350)
    p.add_argument("--crossover-rate", type=float, default=0.95)
    p.add_argument("--mutation-rate", type=float, default=0.15)
    p.add_argument("--tournament-k", type=int, default=3)
    p.add_argument("--elite-size", type=int, default=3)
    p.add_argument("--stall-generations", type=int, default=120)
    p.add_argument("--seed-greedy-fraction", type=float, default=0.05)
    p.add_argument("--random-baseline-samples", type=int, default=250)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
