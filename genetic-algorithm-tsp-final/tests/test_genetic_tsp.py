import random

import numpy as np

from src.genetic_tsp import (
    GAConfig,
    GeneticTSPSolver,
    euclidean_distance_matrix,
    ipmx_crossover,
    is_valid_tour,
    linear_mutation,
    tour_length,
)


def test_ipmx_matches_paper_example():
    p1 = [1, 2, 3, 4, 5, 6, 7, 8]
    p2 = [2, 7, 5, 8, 4, 1, 6, 3]
    c1, c2 = ipmx_crossover(p1, p2, 2, 6)
    assert c1 == [6, 2, 1, 4, 5, 8, 7, 3]
    assert c2 == [2, 7, 3, 4, 5, 6, 1, 8]


def test_linear_mutation_matches_paper_example():
    parent = [2, 3, 5, 6, 1, 4, 7, 8]
    assert linear_mutation(parent, 8) == [3, 4, 6, 7, 2, 5, 8, 1]


def test_ipmx_preserves_permutations_property():
    rng = random.Random(123)
    for n in range(4, 15):
        base = list(range(1, n + 1))
        for _ in range(50):
            p1, p2 = base[:], base[:]
            rng.shuffle(p1)
            rng.shuffle(p2)
            a, b = sorted(rng.sample(range(0, n + 1), 2))
            if a == b:
                continue
            c1, c2 = ipmx_crossover(p1, p2, a, b)
            assert is_valid_tour(c1, n)
            assert is_valid_tour(c2, n)


def test_tour_length_square():
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    dist = euclidean_distance_matrix(coords)
    assert np.isclose(tour_length([1, 2, 3, 4], dist), 4.0)


def test_solver_is_reproducible():
    rng = np.random.default_rng(7)
    coords = rng.random((12, 2)) * 100
    dist = euclidean_distance_matrix(coords)
    cfg = GAConfig(population_size=60, generations=120, stall_generations=40, random_seed=19)
    result1 = GeneticTSPSolver(dist, cfg).solve()
    result2 = GeneticTSPSolver(dist, cfg).solve()
    assert result1[0] == result2[0]
    assert np.isclose(result1[1], result2[1])
    assert result1[2] == result2[2]
