# GitHub setup

Recommended repository name:

`genetic-algorithm-tsp`

Recommended description:

> Reproducible Genetic Algorithm solver for the Traveling Salesman Problem using IPMX crossover, linear mutation, greedy seeding, elitism, and convergence analysis.

Recommended topics:

`genetic-algorithm` `tsp` `optimization` `evolutionary-algorithms` `python` `numpy` `operations-research` `metaheuristics`

## GitHub About / description options

### Recommended GitHub description

`Reproducible Genetic Algorithm solver for the Traveling Salesman Problem with IPMX crossover, mutation, baselines, tests, and convergence analysis.`

### Shorter alternative

`Genetic Algorithm for TSP with IPMX crossover, reproducible experiments, baselines, tests, and visualizations.`

### GitHub profile / pinned-project blurb

`A from-scratch Genetic Algorithm implementation for the Traveling Salesman Problem, featuring IPMX crossover, linear mutation, tournament selection, elitism, deterministic benchmarking, and route/convergence visualizations.`

### LinkedIn project description

`Implemented and evaluated a reproducible Genetic Algorithm for the Traveling Salesman Problem. The project includes custom permutation operators, tournament selection, elitism, greedy initialization, early stopping, deterministic experiments, automated tests, baseline comparisons, and visualizations of both the optimized route and convergence behavior.`

### Portfolio one-liner

`Built a reproducible evolutionary-optimization pipeline for TSP and benchmarked it against random and nearest-neighbor baselines.`


## Publish from PowerShell

Create an empty **public** repository on GitHub named `genetic-algorithm-tsp`. Do not initialize it with a README, license, or .gitignore because they are already included here.

Then run from this project directory:

```powershell
git init
git add .
git commit -m "Initial commit: reproducible genetic algorithm TSP solver"
git branch -M main
git remote add origin https://github.com/Amirreza-Naseri/genetic-algorithm-tsp.git
git push -u origin main
```

If `origin` already exists:

```powershell
git remote set-url origin https://github.com/Amirreza-Naseri/genetic-algorithm-tsp.git
git push -u origin main
```

## Suggested CV bullet

**Genetic Algorithm for Traveling Salesman Problem** — Implemented a reproducible permutation-based GA with IPMX crossover, paper-specified linear mutation, tournament selection, elitism, greedy initialization, early stopping, automated tests, and convergence/route visualizations; benchmarked the method against random-tour and nearest-neighbor baselines on a deterministic 30-city Euclidean instance.
