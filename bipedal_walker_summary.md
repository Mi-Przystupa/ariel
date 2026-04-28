# Bipedal Walker Evolution (examples/re_book/4_bipedal_walker_evolution.py)

## Purpose
Evolve bipedal robot morphologies and neural controllers. Morphology is a tree
genome constrained to exactly two legs; controllers are feedforward NNs
optimized by an inner CMA-ES loop (Lamarckian inheritance).

## Morphology (Tree Genome)
- Core has exactly two children: LEFT and RIGHT legs.
- Legs are structured hinge–brick chains with alternating hinge rotations
  (0/90°) to allow multi-plane bending.
- Constraints: minimum leg length and minimum hinges per leg.
- Mutations are rolled back if constraints break.

## Controller (Neural Network)
- 2-hidden-layer MLP with ELU activations and tanh outputs scaled to joint
  limits.
- Input: robot state + phase signals (sin/cos).
- Parameters are learned per individual via CMA-ES.

## Fitness
Composite minimization objective:
- Strong reward for standing height above resting.
- Forward distance reward only if standing.
- Combined height × distance bonus.
- Velocity penalty for instability.
- Crash penalty if core contacts floor.

## Evolution Loop (μ+λ)
- Initialize population with valid bipedal morphologies.
- For each generation:
  1. Parent selection (top μ).
  2. Reproduction (crossover + safe mutations).
  3. Inner learning (CMA-ES) for controllers.
  4. Evaluation (simulation fitness).
  5. Survivor selection (top μ of μ+λ).
  6. Diagnostics + video export of top 3.

## Simulation
- MuJoCo SimpleFlatWorld.
- Early termination on core-floor contact.
- Optional visualization of best individual.
