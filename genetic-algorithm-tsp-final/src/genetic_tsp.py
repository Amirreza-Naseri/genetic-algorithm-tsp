from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


def ipmx_crossover(
    p1: Sequence[int],
    p2: Sequence[int],
    cut1: int,
    cut2: int,
) -> Tuple[List[int], List[int]]:
    """Increasing Partially Mapped Crossover (IPMX).

    The implementation follows the operator described by Sharma et al. (2024):
    sort the segments between two cut points, swap the sorted segments, keep
    non-conflicting genes from the primary parent, then resolve conflicts by
    partial mapping.
    """
    p1, p2 = list(p1), list(p2)
    n = len(p1)
    if n != len(p2):
        raise ValueError("Parents must have the same length.")
    if n < 2:
        raise ValueError("Chromosomes must contain at least two genes.")
    if set(p1) != set(p2) or len(set(p1)) != n:
        raise ValueError("Parents must be permutations of the same unique genes.")
    if not (0 <= cut1 <= n and 0 <= cut2 <= n) or cut1 == cut2:
        raise ValueError("Cuts must be distinct and within [0, n].")

    i, j = sorted((cut1, cut2))
    if not (0 <= i < j <= n):
        raise ValueError("Invalid cut points.")

    seg1_sorted = sorted(p1[i:j])
    seg2_sorted = sorted(p2[i:j])

    c1: List[Optional[int]] = [None] * n
    c2: List[Optional[int]] = [None] * n
    c1[i:j] = seg2_sorted
    c2[i:j] = seg1_sorted

    used1, used2 = set(seg2_sorted), set(seg1_sorted)
    pos_in_p2 = {g: idx for idx, g in enumerate(p2)}
    pos_in_p1 = {g: idx for idx, g in enumerate(p1)}

    for k in range(n):
        if i <= k < j:
            continue
        if p1[k] not in used1:
            c1[k] = p1[k]
            used1.add(p1[k])
        if p2[k] not in used2:
            c2[k] = p2[k]
            used2.add(p2[k])

    for k in range(n):
        if c1[k] is None:
            g = p1[k]
            seen = set()
            while g in used1:
                if g in seen:
                    g = next(x for x in p1 if x not in used1)
                    break
                seen.add(g)
                g = p1[pos_in_p2[g]]
            c1[k] = g
            used1.add(g)

        if c2[k] is None:
            g = p2[k]
            seen = set()
            while g in used2:
                if g in seen:
                    g = next(x for x in p2 if x not in used2)
                    break
                seen.add(g)
                g = p2[pos_in_p1[g]]
            c2[k] = g
            used2.add(g)

    return [int(x) for x in c1], [int(x) for x in c2]  # type: ignore[arg-type]


def linear_mutation(chromosome: Sequence[int], n_cities: int) -> List[int]:
    """Paper-specified linear mutation L(x_i)=x_i+1 with wrap-around."""
    if n_cities <= 0:
        raise ValueError("n_cities must be positive.")
    chromosome = list(chromosome)
    if set(chromosome) != set(range(1, n_cities + 1)):
        raise ValueError("Chromosome must be a 1-based permutation of city labels.")
    return [(gene % n_cities) + 1 for gene in chromosome]


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 2:
        raise ValueError("coords must be a 2D array with at least two cities.")
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def tour_length(tour_1based: Sequence[int], dist: np.ndarray) -> float:
    tour = list(tour_1based)
    n = len(tour)
    if not is_valid_tour(tour, n):
        raise ValueError("tour must be a valid 1-based permutation.")
    idx = np.asarray(tour, dtype=int) - 1
    next_idx = np.roll(idx, -1)
    return float(dist[idx, next_idx].sum())


def is_valid_tour(tour: Sequence[int], n: int) -> bool:
    return len(tour) == n and set(tour) == set(range(1, n + 1))


def greedy_nearest_neighbor(dist: np.ndarray, start: int = 0) -> List[int]:
    """Nearest-neighbor heuristic. `start` is a zero-based city index."""
    n = int(dist.shape[0])
    if not (0 <= start < n):
        raise ValueError("start must be a valid zero-based city index.")
    unvisited = set(range(n))
    cur = start
    tour = [cur]
    unvisited.remove(cur)
    while unvisited:
        nxt = min(unvisited, key=lambda j: (dist[cur, j], j))
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return [x + 1 for x in tour]


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 120
    generations: int = 350
    crossover_rate: float = 0.95
    mutation_rate: float = 0.15
    tournament_k: int = 3
    elite_size: int = 3
    stall_generations: int = 120
    seed_greedy_fraction: float = 0.05
    random_seed: Optional[int] = 7

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.generations < 1:
            raise ValueError("generations must be positive.")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("crossover_rate must be in [0, 1].")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0, 1].")
        if not 0 <= self.seed_greedy_fraction <= 1:
            raise ValueError("seed_greedy_fraction must be in [0, 1].")
        if not 1 <= self.tournament_k <= self.population_size:
            raise ValueError("tournament_k must be between 1 and population_size.")
        if not 0 <= self.elite_size < self.population_size:
            raise ValueError("elite_size must be in [0, population_size).")
        if self.stall_generations < 1:
            raise ValueError("stall_generations must be positive.")


class GeneticTSPSolver:
    """Genetic algorithm solver for symmetric TSP distance matrices."""

    def __init__(self, dist: np.ndarray, config: GAConfig = GAConfig()):
        self.dist = np.asarray(dist, dtype=float)
        if self.dist.ndim != 2 or self.dist.shape[0] != self.dist.shape[1]:
            raise ValueError("dist must be a square matrix.")
        if self.dist.shape[0] < 2:
            raise ValueError("At least two cities are required.")
        self.n = self.dist.shape[0]
        self.cfg = config
        self.cfg.validate()
        self.rng = random.Random(self.cfg.random_seed)

    def _random_tour(self) -> List[int]:
        tour = list(range(1, self.n + 1))
        self.rng.shuffle(tour)
        return tour

    def _init_population(self) -> List[List[int]]:
        pop: List[List[int]] = []
        greedy_count = int(round(self.cfg.population_size * self.cfg.seed_greedy_fraction))
        for _ in range(greedy_count):
            start = self.rng.randrange(self.n)
            pop.append(greedy_nearest_neighbor(self.dist, start=start))
        while len(pop) < self.cfg.population_size:
            pop.append(self._random_tour())
        return pop

    def _evaluate(self, pop: Sequence[Sequence[int]]) -> np.ndarray:
        return np.asarray([tour_length(ind, self.dist) for ind in pop], dtype=float)

    def _tournament_select(self, pop: List[List[int]], scores: np.ndarray) -> List[int]:
        idxs = self.rng.sample(range(len(pop)), self.cfg.tournament_k)
        best = min(idxs, key=lambda i: scores[i])
        return pop[best][:]

    def solve(self) -> Tuple[List[int], float, List[float]]:
        pop = self._init_population()
        scores = self._evaluate(pop)

        best_idx = int(np.argmin(scores))
        best_tour = pop[best_idx][:]
        best_len = float(scores[best_idx])
        history_best = [best_len]
        stall = 0

        for _ in range(self.cfg.generations):
            elite_idxs = np.argsort(scores)[: self.cfg.elite_size]
            new_pop = [pop[int(i)][:] for i in elite_idxs]

            while len(new_pop) < self.cfg.population_size:
                p1 = self._tournament_select(pop, scores)
                p2 = self._tournament_select(pop, scores)
                for _ in range(10):
                    if p2 != p1:
                        break
                    p2 = self._tournament_select(pop, scores)

                if self.rng.random() < self.cfg.crossover_rate:
                    cut1 = self.rng.randrange(self.n)
                    cut2 = self.rng.randrange(self.n)
                    while cut2 == cut1:
                        cut2 = self.rng.randrange(self.n)
                    c1, c2 = ipmx_crossover(p1, p2, cut1, cut2)
                else:
                    c1, c2 = p1[:], p2[:]

                if self.rng.random() < self.cfg.mutation_rate:
                    c1 = linear_mutation(c1, self.n)
                if self.rng.random() < self.cfg.mutation_rate:
                    c2 = linear_mutation(c2, self.n)

                if not is_valid_tour(c1, self.n) or not is_valid_tour(c2, self.n):
                    raise RuntimeError("Genetic operator produced an invalid tour.")

                new_pop.append(c1)
                if len(new_pop) < self.cfg.population_size:
                    new_pop.append(c2)

            pop = new_pop
            scores = self._evaluate(pop)
            cur_best_idx = int(np.argmin(scores))
            cur_best_len = float(scores[cur_best_idx])

            if cur_best_len + 1e-12 < best_len:
                best_len = cur_best_len
                best_tour = pop[cur_best_idx][:]
                stall = 0
            else:
                stall += 1

            history_best.append(best_len)
            if stall >= self.cfg.stall_generations:
                break

        return best_tour, best_len, history_best
