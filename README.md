# Genetic Algorithm for the Traveling Salesman Problem

A reproducible Python implementation of a permutation-based **Genetic Algorithm (GA)** for the **Traveling Salesman Problem (TSP)**. The project implements the **Increasing Partially Mapped Crossover (IPMX)** and the paper-specified **linear mutation** operator, then extends them with a complete optimization pipeline: tournament selection, elitism, greedy population seeding, early stopping, deterministic experiments, automated tests, baselines, and visualizations.

> **Portfolio focus:** evolutionary optimization, algorithm implementation, experimental reproducibility, testing, and result communication.

## Project overview

This project explores how a Genetic Algorithm can solve a combinatorial optimization problem where the goal is to find a short closed route through a set of cities. Rather than relying only on library-level optimization APIs, the core evolutionary operators are implemented explicitly so the search process is transparent and testable.

The repository is designed as a portfolio project: it combines algorithm implementation with reproducible experimentation, baseline comparison, automated testing, command-line execution, and visual result reporting. It is especially relevant to roles involving data science, optimization, operations research, or machine learning engineering.

**Key skills demonstrated:** Python, NumPy, evolutionary algorithms, combinatorial optimization, experimental design, reproducibility, unit testing, benchmarking, and data visualization.

## Demo results

The repository includes a deterministic 30-city Euclidean benchmark generated with seed `7`. The default GA configuration uses 120 individuals, 350 maximum generations, 5% greedy seeding, a 0.95 crossover rate, a 0.15 mutation rate, elitism of 3, and early stopping after 120 stagnant generations. Running the default experiment reproduces the CSV files and plots in `results/`.

| Metric | Value |
|---|---:|
| Cities | 30 |
| Random-tour mean | `1600.994` |
| Nearest-neighbor baseline | `583.350` |
| GA best tour | `546.627` |
| Improvement vs. greedy | `6.30%` |
| Generations executed | `202` |

The GA improved the fixed-start nearest-neighbor baseline by **6.30%** on this benchmark. Its best-so-far distance decreased from **561.491** in the initial population to **546.627** before early stopping. This is a heuristic result, **not a claim of the global optimum**.

### Best route

![Best route](genetic-algorithm-tsp-final
/results/images/best_route.png)

### Convergence

![Convergence](genetic-algorithm-tsp-final/
results/images/convergence.png)

### Baseline comparison

![Baseline comparison](genetic-algorithm-tsp-final/
results/images/baseline_comparison.png)

## Algorithm

Each chromosome is a permutation of city IDs. The solver uses:

- **Population initialization:** random tours plus nearest-neighbor seeded tours
- **Selection:** tournament selection
- **Crossover:** Increasing Partially Mapped Crossover (IPMX)
- **Mutation:** linear mutation `L(x_i) = x_i + 1`, wrapping `n -> 1`
- **Survival:** elitism
- **Stopping:** maximum generations or no improvement for a configured number of generations
- **Objective:** minimize the total closed-tour Euclidean distance

## Reference implementation target

The IPMX and linear mutation operators are based on:

> Sharma, M. K., Chaudhary, S., Rathour, L., & Mishra, V. N. (2024). *Modified genetic algorithm with novel crossover and mutation operator for travelling salesman problem*. Sigma Journal of Engineering and Natural Sciences, 42(6), 1876–1883. DOI: `10.14744/sigma.2023.00105`.

The paper's IPMX example is encoded as an automated unit test. This repository is an independent implementation and adds its own solver structure, reproducibility controls, baselines, tests, CLI, and visualizations.

## Project structure

```text
genetic-algorithm-tsp/
├── .github/workflows/tests.yml
├── data/
│   └── demo_cities.csv
├── results/
│   ├── benchmark_summary.csv
│   ├── best_tour.csv
│   ├── convergence.csv
│   └── images/
│       ├── baseline_comparison.png
│       ├── best_route.png
│       └── convergence.png
├── src/
│   ├── genetic_tsp.py
│   └── run_experiment.py
├── tests/
│   └── test_genetic_tsp.py
├── CITATION.md
├── GITHUB_SETUP.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Reproduce the experiment

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
python src/run_experiment.py
```

Run the tests:

```bash
pytest -q
```

To run a custom coordinate file:

```bash
python src/run_experiment.py --coords data/my_cities.csv
```

The coordinate CSV should contain:

```csv
city,x,y
1,12.4,71.0
2,25.1,19.8
```

## Reproducibility improvement over the original script

The original demo generated city coordinates **before** seeding the random number generator, so `random_seed=7` did not make the complete experiment reproducible. In this version, the city coordinates are generated with `numpy.random.default_rng(seed)`, and the GA uses its own local seeded random-number generator. Running the same configuration produces the same benchmark and tour.

## Engineering choices

The project separates algorithm code from experiment code, validates chromosomes and configuration values, checks that genetic operators preserve legal permutations, writes machine-readable CSV outputs, and uses a CI workflow to run tests on every push or pull request.

## Limitations

- Genetic Algorithms are stochastic heuristics and do not guarantee a globally optimal TSP solution.
- The linear mutation used here follows the referenced paper and changes every city label in a chromosome; it should be interpreted as a paper-specific operator rather than a conventional local TSP mutation such as swap or inversion.
- The included benchmark is a synthetic Euclidean instance. Real-world routing would need road-network distances, constraints, and potentially asymmetric costs.

## License

MIT. See `LICENSE`.
