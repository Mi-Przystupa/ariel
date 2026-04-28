#!/bin/bash
# Run bipedal walker evolution.
#
# Cost per generation ≈ (pop + lambda) × lr_budget × lr_pop × sim_time
# With defaults below: (10 + 20) × 10 × 5 × 5s = 7500 simulations/gen
# Each sim ~0.1s → ~12 min/gen with 1 worker, ~1.5 min/gen with 8 workers.

# --- Evolution ---
GENERATIONS=30       # number of EA generations
POPULATION=10        # population size (mu), lambda = 2x this
DURATION=5           # simulation duration in seconds

# --- Learning (inner CMA-ES per individual) ---
LR_BUDGET=10         # CMA-ES steps per individual
LR_POP=5             # CMA-ES internal population size

# --- Parallelization ---
WORKERS=$(nproc)     # number of parallel workers (default: all CPUs)

# --- Visualization ---
VISUALIZE=true       # launch MuJoCo viewer for best individual at the end

# Build the flag
if [ "$VISUALIZE" = true ]; then
    VIS_FLAG="--visualize"
else
    VIS_FLAG="--no-visualize"
fi

uv run examples/re_book/4_bipedal_walker_evolution.py \
    --budget "$GENERATIONS" \
    --pop "$POPULATION" \
    --dur "$DURATION" \
    --lr-budget "$LR_BUDGET" \
    --lr-pop "$LR_POP" \
    --workers "$WORKERS" \
    "$VIS_FLAG"
