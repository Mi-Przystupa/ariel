"""Bipedal Walker Evolution.

Evolves bipedal walkers using:
  - Tree Genome for morphology (exactly 2 limbs from core)
  - Neural Network controller optimized by inner CMA-ES (Lamarckian)
  - mu+lambda evolutionary strategy
  - Composite fitness: forward distance, head height, stability, crash penalty
"""

# Standard library
import argparse
import copy
import logging
import multiprocessing as mp
import random
import warnings
from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco
import numpy as np
import torch
from evotorch.algorithms import CMAES
from evotorch.neuroevolution import NEProblem
from mujoco import viewer
from rich.console import Console
from rich.progress import track
from rich.traceback import install
from torch import nn

# Silence noisy EvoTorch/torch logs
logging.getLogger("evotorch").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

# ARIEL imports
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.ec import EA, EAOperation, EASettings, Individual, Population
from ariel.ec.genotypes.tree.operators import (
    add_node,
    crossover_subtree,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.simulation.controllers.utils.data_get import (
    get_state_from_data as get_robot_state,
)
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import video_renderer
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# Rich setup
install()
console = Console()

# ============================================================================ #
#                               CONFIGURATION                                  #
# ============================================================================ #

parser = argparse.ArgumentParser(
    description="Bipedal Walker Evolution (Tree Genome + NN + CMA-ES)",
)
parser.add_argument(
    "--budget", type=int, default=30, help="Number of EA generations",
)
parser.add_argument(
    "--pop", type=int, default=20, help="Population size (mu)",
)
parser.add_argument(
    "--dur", type=int, default=15, help="Simulation duration (seconds)",
)
parser.add_argument(
    "--lr-budget",
    type=int,
    default=20,
    help="CMA-ES learning budget per individual",
)
parser.add_argument(
    "--lr-pop",
    type=int,
    default=10,
    help="CMA-ES internal population size",
)
parser.add_argument(
    "--workers",
    type=int,
    default=mp.cpu_count(),
    help="Number of parallel workers (default: all CPUs)",
)
parser.add_argument(
    "--visualize",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Launch MuJoCo viewer for best individual",
)
args = parser.parse_args()

# Constants
DURATION: int = args.dur
MU: int = args.pop
LAMBDA: int = MU * 2
BUDGET: int = args.budget
LEARNING_BUDGET: int = args.lr_budget
LEARNING_POP: int = args.lr_pop
NUM_WORKERS: int = args.workers
MAX_MODULES_PER_LEG: int = 10
HIDDEN_SIZE: int = 32
Z_CRASH_THRESHOLD: float = 0.05
CONTROL_STEP_FREQ: int = 50

SPAWN_POSITION = (0.0, 0.0, 0.1)

# Generation counter (mutable, incremented by diagnostics)
GENERATION = 0

# Determinism
SEED = 42
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Type aliases
type ViewerTypes = Literal["simple", "launcher", "video"]

# Data paths
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)


# ============================================================================ #
#                          NEURAL NETWORK CONTROLLER                           #
# ============================================================================ #


class Network(nn.Module):
    """Feedforward NN controller for joint actuation."""

    def __init__(
        self, input_size: int, output_size: int, hidden_size: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()

        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self, state: np.ndarray) -> np.ndarray:
        x = torch.tensor(state, dtype=torch.float32)
        x = self.hidden_activation(self.fc1(x))
        x = self.hidden_activation(self.fc2(x))
        x = self.output_activation(self.fc3(x)) * (torch.pi / 2)
        return x.detach().numpy()


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: torch.Tensor) -> None:
    """Fill NN parameters from a flat 1-D vector."""
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(
            vector[address : address + n], device=d.device,
        )
        address += n
    if address != len(vector):
        msg = "Parameter vector size mismatch"
        raise IndexError(msg)


def count_parameters(net: nn.Module) -> int:
    """Count total trainable parameters in a network."""
    return sum(p.numel() for p in net.parameters())


# ============================================================================ #
#                       BIPEDAL TREE GENOME HELPERS                            #
# ============================================================================ #


def random_bipedal_tree(max_modules_per_leg: int = MAX_MODULES_PER_LEG) -> TreeGenome:
    """Generate a tree genome with exactly 2 limbs from core (LEFT, RIGHT).

    Each leg is a chain of alternating HINGE and BRICK modules.
    """
    g = TreeGenome()
    g.nodes = {IDX_OF_CORE: {"type": "CORE", "rotation": "DEG_0"}}
    g.edges = []

    leg_faces = ["LEFT", "RIGHT"]
    next_id = 1

    for face in leg_faces:
        leg_length = random.randint(2, max_modules_per_leg)
        parent_id = IDX_OF_CORE
        parent_face = face

        for i in range(leg_length):
            # Alternate HINGE and BRICK for articulated legs
            if i % 2 == 0:
                mtype = "HINGE"
            else:
                mtype = "BRICK"

            rotations = [
                r.name for r in ALLOWED_ROTATIONS[ModuleType[mtype]]
            ]
            rot = random.choice(rotations)

            add_node(g, parent_id, parent_face, next_id, mtype, rot)
            parent_id = next_id
            # HINGE only has FRONT; BRICK can chain via FRONT
            parent_face = "FRONT"
            next_id += 1

    return g


def is_bipedal(genome: TreeGenome) -> bool:
    """Check that the root node has exactly 2 children."""
    core_children = [
        e for e in genome.edges if e["parent"] == IDX_OF_CORE
    ]
    return len(core_children) == 2


MIN_MODULES_PER_LEG = 3   # at least hinge-brick-hinge per leg
MIN_HINGES_PER_LEG = 2    # need at least 2 joints to walk


def validate_morphology(genome: TreeGenome) -> str | None:
    """Check morphology constraints. Returns None if valid, else reason."""
    if not is_bipedal(genome):
        return "not bipedal"

    g = genome.to_networkx()

    # Get the two legs (subtrees from core children)
    core_children = [e["child"] for e in genome.edges if e["parent"] == IDX_OF_CORE]
    if len(core_children) != 2:
        return "not bipedal"

    import networkx as nx
    for i, child in enumerate(core_children):
        # Count modules in this leg (child + descendants)
        leg_nodes = {child} | set(nx.descendants(g, child))
        leg_size = len(leg_nodes)

        if leg_size < MIN_MODULES_PER_LEG:
            return f"leg {i+1} too short ({leg_size} < {MIN_MODULES_PER_LEG} modules)"

        # Count hinges in this leg
        hinges = sum(
            1 for n in leg_nodes
            if genome.nodes[n]["type"] == "HINGE"
        )
        if hinges < MIN_HINGES_PER_LEG:
            return f"leg {i+1} too few joints ({hinges} < {MIN_HINGES_PER_LEG} hinges)"

    return None


def validate_spawned_robot(ind: Individual) -> str | None:
    """Spawn robot and check it doesn't immediately crash (no-control test).

    Returns None if OK, else a reason string.
    """
    robot = genotype_to_phenotype(ind)
    if robot is None:
        return "failed to build"

    try:
        world = SimpleFlatWorld(load_precompiled=False)
        world.spawn(
            robot.spec,
            position=SPAWN_POSITION,
            correct_collision_with_floor=True,
        )
        model = world.spec.compile()
        data = mujoco.MjData(model)
    except Exception:
        return "failed to compile"

    if model.nu == 0:
        return "no actuators"

    # Run for 0.5s with zero control to check passive stability
    mujoco.mj_resetData(model, data)
    steps = int(0.5 / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)

    core_z = data.qpos[2]
    if core_z < Z_CRASH_THRESHOLD:
        return f"core hits ground passively (z={core_z:.3f})"

    return None


def bipedal_safe_mutate(genome: TreeGenome, mutation_fn) -> None:
    """Apply a mutation, rolling back if morphology constraints break."""
    old_nodes = copy.deepcopy(genome.nodes)
    old_edges = copy.deepcopy(genome.edges)
    mutation_fn(genome)
    if validate_morphology(genome) is not None:
        genome.nodes = old_nodes
        genome.edges = old_edges


# ============================================================================ #
#                       GENOTYPE-TO-PHENOTYPE PIPELINE                         #
# ============================================================================ #


def genotype_to_phenotype(ind: Individual):
    """Decode tree genome into a MuJoCo robot spec.

    Returns
    -------
    CoreModule or None
    """
    genome = TreeGenome.from_dict(ind.genotype["morph"])
    graph = genome.to_networkx()
    if graph.number_of_nodes() == 0:
        return None
    try:
        return construct_mjspec_from_graph(graph)
    except Exception:
        return None


def get_joint_count(ind: Individual) -> int:
    """Return number of actuators for an individual's morphology."""
    robot = genotype_to_phenotype(ind)
    if robot is None:
        return 0
    try:
        model = robot.spec.compile()
        return model.nu
    except Exception:
        return 0


# ============================================================================ #
#                           FITNESS FUNCTION                                   #
# ============================================================================ #


def bipedal_fitness(
    trajectory: list,
) -> float:
    """Composite fitness for bipedal walker (minimization: lower is better).

    Components
    ----------
    1. Hard crash: core z drops below crash threshold → +10 penalty
    2. Forward distance: reward moving forward (negative = good)
    3. Head height: reward being tall (core lifted off ground)
    4. Head height × forward distance: reward tall + moving
    5. Head velocity penalty: penalize jerky/unstable heads

    The fitness is designed so that CMA-ES always has a gradient:
    - Standing still at z=0.10 scores ~0 (neutral)
    - Lifting core higher scores negative (good)
    - Moving forward scores negative (good)
    - Crashing scores +10 (bad)
    """
    if not trajectory or len(trajectory) < 2:
        return 999.0

    positions = np.array(trajectory)

    # Hard crash: core hit the floor
    min_z = positions[:, 2].min()
    if min_z < Z_CRASH_THRESHOLD:
        return 10.0

    # Forward distance (x-axis)
    forward_dist = positions[-1, 0] - positions[0, 0]

    # Head height reward: how high is the core above resting position?
    # Resting z ≈ 0.10 (core sitting on floor). Reward lifting above this.
    resting_z = 0.10
    avg_height_above_rest = max(positions[:, 2].mean() - resting_z, 0.0)

    # Height × forward distance (reward being tall AND moving)
    height_reward = avg_height_above_rest * max(forward_dist, 0.0)

    # Head velocity penalty (stability)
    velocities = np.diff(positions, axis=0)
    head_speed = np.linalg.norm(velocities, axis=1)
    velocity_penalty = head_speed.mean() * 0.1

    # Pure height reward: even if not moving, reward standing tall
    height_only = avg_height_above_rest * 5.0

    # Minimize: lower is better
    fitness = -forward_dist - height_reward - height_only + velocity_penalty
    return float(fitness)


# ============================================================================ #
#                          SIMULATION RUNNER                                    #
# ============================================================================ #


def run_simulation(
    ind: Individual,
    mode: ViewerTypes = "simple",
) -> float:
    """Build phenotype, run simulation, return fitness."""
    mujoco.set_mjcb_control(None)

    # 1. Build body
    robot = genotype_to_phenotype(ind)
    if robot is None:
        return 999.0

    try:
        test_model = robot.spec.compile()
    except Exception:
        return 999.0

    if test_model.nu == 0:
        return 999.0
    del test_model

    # 2. Setup environment
    world = SimpleFlatWorld(load_precompiled=False)
    world.spawn(
        robot.spec,
        position=SPAWN_POSITION,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # 3. Setup NN controller
    num_joints = len(data.qpos) - 7  # exclude free joint (3 pos + 4 quat)
    input_dim = 3 + num_joints + 2  # quat_imag + joints + heartbeat
    network = Network(
        input_size=input_dim,
        output_size=model.nu,
        hidden_size=HIDDEN_SIZE,
    )

    # Load learned brain weights if available
    brain_weights = ind.tags.get("brain")
    if brain_weights is not None:
        expected_params = count_parameters(network)
        if len(brain_weights) == expected_params:
            fill_parameters(network, torch.tensor(brain_weights))

    # 4. Setup tracker
    tracker = Tracker(mujoco.mjtObj.mjOBJ_BODY, "core", ["xpos"])
    tracker.setup(world.spec, data)

    # 5. Run simulation
    mujoco.mj_resetData(model, data)

    if mode == "simple":
        _run_headless(model, data, network, tracker)
    elif mode == "launcher":
        # For viewer, use control callback
        current_ctrl = np.zeros(model.nu)

        def control_callback(m, d):
            nonlocal current_ctrl
            deduced_step = int(np.ceil(d.time / m.opt.timestep))
            if deduced_step % CONTROL_STEP_FREQ == 0:
                robot_state = get_robot_state(d)
                phase = [
                    2 * np.sin(d.time * 2.0 * np.pi),
                    2 * np.cos(d.time * 2.0 * np.pi),
                ]
                state = np.concatenate([
                    robot_state, phase,
                ]).astype(np.float32)
                current_ctrl = network.forward(state)
            d.ctrl[:] = current_ctrl

        mujoco.set_mjcb_control(control_callback)
        viewer.launch(model=model, data=data)
        return 0.0  # No fitness for interactive viewer
    elif mode == "video":
        _run_headless(model, data, network, tracker)
        # Re-run for video
        mujoco.mj_resetData(model, data)
        current_ctrl = np.zeros(model.nu)

        def video_ctrl(m, d):
            nonlocal current_ctrl
            deduced_step = int(np.ceil(d.time / m.opt.timestep))
            if deduced_step % CONTROL_STEP_FREQ == 0:
                robot_state = get_robot_state(d)
                phase = [
                    2 * np.sin(d.time * 2.0 * np.pi),
                    2 * np.cos(d.time * 2.0 * np.pi),
                ]
                state = np.concatenate([
                    robot_state, phase,
                ]).astype(np.float32)
                current_ctrl = network.forward(state)
            d.ctrl[:] = current_ctrl

        mujoco.set_mjcb_control(video_ctrl)
        recorder = VideoRecorder(
            output_folder=str(DATA / "videos"),
            file_name=f"bipedal_{ind.id}",
        )
        video_renderer(model, data, duration=DURATION, video_recorder=recorder)

    # 6. Compute fitness from trajectory
    if not tracker.history["xpos"]:
        return 999.0

    first_key = next(iter(tracker.history["xpos"].keys()))
    traj = tracker.history["xpos"][first_key]
    return bipedal_fitness(traj)


def _run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    network: Network,
    tracker: Tracker,
) -> bool:
    """Run headless simulation with early termination on head crash.

    Returns True if head crashed.
    """
    timestep = model.opt.timestep
    total_steps = int(DURATION / timestep)
    current_ctrl = np.zeros(model.nu)
    head_crashed = False

    # Warmup: let physics settle for 0.5s before tracking/control
    warmup_steps = int(0.5 / timestep)
    for _ in range(warmup_steps):
        mujoco.mj_step(model, data)

    # Initial tracker reading (after warmup)
    tracker.update(data)

    for step in range(total_steps - warmup_steps):
        # Control step
        if step % CONTROL_STEP_FREQ == 0:
            robot_state = get_robot_state(data)
            phase = [
                2 * np.sin(data.time * 2.0 * np.pi),
                2 * np.cos(data.time * 2.0 * np.pi),
            ]
            state = np.concatenate([
                robot_state, phase,
            ]).astype(np.float32)

            try:
                current_ctrl = network.forward(state)
            except Exception:
                break

            # Update tracker at control frequency
            tracker.update(data)

            # Check head/core z-height (crash = core hit the floor)
            first_key = next(iter(tracker.history["xpos"].keys()))
            traj = tracker.history["xpos"][first_key]
            if traj and traj[-1][2] < Z_CRASH_THRESHOLD:
                head_crashed = True
                break

        data.ctrl[:] = current_ctrl
        mujoco.mj_step(model, data)

    # Final tracker update
    tracker.update(data)
    return head_crashed


# ============================================================================ #
#                          INNER CMA-ES LEARNING                               #
# ============================================================================ #


def learn_brain(
    ind: Individual,
    initial_weights: list[float] | None = None,
) -> list[float] | None:
    """Run inner CMA-ES to optimize NN controller weights.

    Parameters
    ----------
    ind : Individual
        The individual whose morphology to learn a controller for.
    initial_weights : list[float] or None
        Inherited brain weights to warm-start from (Lamarckian).

    Returns
    -------
    list[float] or None
        Best weight vector found, or None if learning failed.
    """
    mujoco.set_mjcb_control(None)

    robot = genotype_to_phenotype(ind)
    if robot is None:
        return None

    # Build in world to get correct dimensions (free joint adds 7 qpos)
    world_check = SimpleFlatWorld(load_precompiled=False)
    try:
        world_check.spawn(
            robot.spec,
            position=SPAWN_POSITION,
            correct_collision_with_floor=True,
        )
        test_model = world_check.spec.compile()
        test_data = mujoco.MjData(test_model)
    except Exception:
        return None

    if test_model.nu == 0:
        return None

    num_joints = len(test_data.qpos) - 7
    input_dim = 3 + num_joints + 2
    output_dim = test_model.nu
    del test_model, test_data, world_check

    # Create network template
    network = Network(
        input_size=input_dim,
        output_size=output_dim,
        hidden_size=HIDDEN_SIZE,
    )

    # Define fitness function for CMA-ES
    def fitness_function(net: Network) -> float:
        mujoco.set_mjcb_control(None)

        robot_inner = genotype_to_phenotype(ind)
        if robot_inner is None:
            return 999.0

        world = SimpleFlatWorld(load_precompiled=False)
        world.spawn(
            robot_inner.spec,
            position=SPAWN_POSITION,
            correct_collision_with_floor=True,
        )
        try:
            model = world.spec.compile()
            data = mujoco.MjData(model)
        except Exception:
            return 999.0

        tracker = Tracker(mujoco.mjtObj.mjOBJ_BODY, "core", ["xpos"])
        tracker.setup(world.spec, data)
        mujoco.mj_resetData(model, data)

        _run_headless(model, data, net, tracker)

        if not tracker.history["xpos"]:
            return 999.0

        first_key = next(iter(tracker.history["xpos"].keys()))
        traj = tracker.history["xpos"][first_key]
        return bipedal_fitness(traj)

    # Setup CMA-ES via EvoTorch
    problem = NEProblem(
        objective_sense="min",
        network_eval_func=fitness_function,
        network=network.eval(),
        initial_bounds=(-0.5, 0.5),
        device="cpu",
    )

    # Warm-start from inherited weights if available and matching size
    center_init: torch.Tensor | None = None
    if initial_weights is not None:
        expected = count_parameters(network)
        if len(initial_weights) == expected:
            center_init = torch.tensor(
                initial_weights, dtype=torch.float32,
            )

    searcher = CMAES(
        problem=problem,
        stdev_init=0.1,
        popsize=LEARNING_POP,
        center_init=center_init,
    )

    # Run learning
    best_fitness = 999.0
    for step in range(LEARNING_BUDGET):
        searcher.step()
        gen_best = searcher.status["pop_best_eval"]
        if gen_best < best_fitness:
            best_fitness = gen_best

    # Extract best weights
    best_weights = searcher.status["best"].values
    return best_weights.tolist()


# ============================================================================ #
#                      PARALLEL WORKER FUNCTIONS                               #
# ============================================================================ #


def _silence_worker_logs() -> None:
    """Suppress EvoTorch/torch noise in worker processes."""
    logging.getLogger("evotorch").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message="To copy construct from a tensor")


def _worker_learn(payload: dict) -> dict:
    """Standalone worker for learning (picklable for multiprocessing).

    Takes and returns plain dicts to avoid SQLModel pickling issues.
    """
    _silence_worker_logs()
    genotype = payload["genotype"]
    inherited_brain = payload["inherited_brain"]

    # Reconstruct a temporary Individual
    ind = Individual()
    ind.genotype = genotype
    brain = learn_brain(ind, initial_weights=inherited_brain)
    return {"brain": brain}


def _worker_evaluate(payload: dict) -> dict:
    """Standalone worker for evaluation (picklable for multiprocessing)."""
    _silence_worker_logs()
    genotype = payload["genotype"]
    brain = payload["brain"]

    ind = Individual()
    ind.genotype = genotype
    ind.tags["brain"] = brain
    fitness = run_simulation(ind, mode="simple")
    return {"fitness": fitness}


# ============================================================================ #
#                            EA OPERATIONS                                     #
# ============================================================================ #


@EAOperation
def parent_selection(population: Population) -> Population:
    """Tag top MU individuals as parents (minimization: lower fitness first)."""
    population = population.sort(sort="min", attribute="fitness_")
    for i, ind in enumerate(population):
        ind.tags["ps"] = i < MU

    ps_count = sum(1 for ind in population if ind.tags.get("ps", False))
    console.log(
        f"[cyan]Parent Selection: {ps_count}/{len(population)} parents[/cyan]",
    )
    return population


@EAOperation
def reproduction(population: Population) -> Population:
    """Create LAMBDA offspring via tree crossover + mutation.

    Lamarckian: children inherit learned brain from fitter parent.
    Bipedal constraint: root node must have exactly 2 children.
    """
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    if not parents:
        console.log("[yellow]Warning: no parents tagged, using all[/yellow]")
        parents = list(population)

    offspring = []
    attempts = 0
    max_attempts = LAMBDA * 10

    while len(offspring) < LAMBDA and attempts < max_attempts:
        attempts += 1

        # Select two parents
        if len(parents) >= 2:
            p1, p2 = random.sample(parents, 2)
        else:
            p1 = p2 = parents[0]

        # Morphology crossover
        morph1 = TreeGenome.from_dict(p1.genotype["morph"])
        morph2 = TreeGenome.from_dict(p2.genotype["morph"])
        child_morph, _ = crossover_subtree(morph1, morph2)

        # Mutate morphology (50% chance each mutation type)
        if random.random() < 0.5:
            bipedal_safe_mutate(child_morph, mutate_replace_node)
        if random.random() < 0.5:
            bipedal_safe_mutate(child_morph, mutate_subtree_replacement)
        if random.random() < 0.3:
            bipedal_safe_mutate(child_morph, mutate_shrink)

        # Validate morphology constraints (bipedal, leg length, hinges)
        reason = validate_morphology(child_morph)
        if reason is not None:
            continue

        try:
            validate_genome_dict(child_morph.to_dict())
        except ValueError:
            continue

        child_ind = Individual()
        child_ind.genotype = {"morph": child_morph.to_dict()}
        child_joints = get_joint_count(child_ind)

        if child_joints == 0:
            continue

        # Check it doesn't immediately crash under gravity
        spawn_reason = validate_spawned_robot(child_ind)
        if spawn_reason is not None:
            continue

        # Lamarckian brain inheritance from fitter parent
        fitter = p1
        if (
            p2.fitness_ is not None
            and (p1.fitness_ is None or p2.fitness_ < p1.fitness_)
        ):
            fitter = p2

        inherited_brain = fitter.tags.get("brain")

        if inherited_brain is not None:
            parent_joints = get_joint_count(fitter)
            if child_joints != parent_joints:
                # NN size mismatch, can't inherit
                inherited_brain = None

        child_ind.tags["brain"] = inherited_brain
        child_ind.tags["requires_lr"] = True
        child_ind.tags["ps"] = False
        child_ind.requires_eval = True

        offspring.append(child_ind)

    population.extend(offspring)
    console.log(
        f"[green]Reproduction: {len(offspring)} offspring created[/green]",
    )
    return population


@EAOperation
def learning(population: Population) -> Population:
    """Inner CMA-ES learning loop for individuals needing brain optimization."""
    to_learn = [
        ind
        for ind in population
        if ind.alive and ind.tags.get("requires_lr", False)
    ]

    if not to_learn:
        return population

    console.log(
        f"[cyan]Learning {len(to_learn)} individuals"
        f" ({NUM_WORKERS} workers)...[/cyan]",
    )

    payloads = [
        {
            "genotype": ind.genotype,
            "inherited_brain": ind.tags.get("brain"),
        }
        for ind in to_learn
    ]

    if NUM_WORKERS > 1 and len(to_learn) > 1:
        with mp.Pool(min(NUM_WORKERS, len(to_learn))) as pool:
            results = pool.map(_worker_learn, payloads)
    else:
        results = [_worker_learn(p) for p in payloads]

    for ind, result in zip(to_learn, results):
        if result["brain"] is not None:
            ind.tags["brain"] = result["brain"]
        ind.tags["requires_lr"] = False

    return population


@EAOperation
def evaluate(population: Population) -> Population:
    """Evaluate all unevaluated individuals."""
    to_eval = [
        ind
        for ind in population
        if ind.alive and ind.requires_eval
    ]

    if not to_eval:
        return population

    console.log(
        f"[cyan]Evaluating {len(to_eval)} individuals"
        f" ({NUM_WORKERS} workers)...[/cyan]",
    )

    payloads = [
        {
            "genotype": ind.genotype,
            "brain": ind.tags.get("brain"),
        }
        for ind in to_eval
    ]

    if NUM_WORKERS > 1 and len(to_eval) > 1:
        with mp.Pool(min(NUM_WORKERS, len(to_eval))) as pool:
            results = pool.map(_worker_evaluate, payloads)
    else:
        results = [_worker_evaluate(p) for p in payloads]

    for ind, result in zip(to_eval, results):
        ind.fitness = result["fitness"]
        ind.requires_eval = False

    # Print fitness summary
    fitnesses = [r["fitness"] for r in results]
    crashed = sum(1 for f in fitnesses if f == 10.0)
    invalid = sum(1 for f in fitnesses if f == 999.0)
    valid = [f for f in fitnesses if f not in (10.0, 999.0)]

    console.log(
        f"[blue]Fitness: "
        f"best={min(fitnesses):.4f}, "
        f"avg={np.mean(fitnesses):.4f}, "
        f"worst={max(fitnesses):.4f} | "
        f"crashed={crashed}, invalid={invalid}, "
        f"walking={len(valid)}[/blue]",
    )
    if valid:
        console.log(
            f"[green]Walking fitnesses: "
            f"best={min(valid):.4f}, avg={np.mean(valid):.4f}[/green]",
        )

    return population


@EAOperation
def survivor_selection(population: Population) -> Population:
    """Keep top MU individuals from parents + offspring (mu+lambda)."""
    population = population.sort(sort="min", attribute="fitness_")
    for i, ind in enumerate(population):
        ind.alive = i < MU
    return population


@EAOperation
def diagnostics(population: Population) -> Population:
    """Print a per-generation summary and save videos of top 3."""
    global GENERATION  # noqa: PLW0603

    survivors = [ind for ind in population if ind.alive]
    all_fitnesses = [
        ind.fitness_ for ind in survivors if ind.fitness_ is not None
    ]

    if not all_fitnesses:
        return population

    # Fitness breakdown
    crashed = [f for f in all_fitnesses if f == 10.0]
    invalid = [f for f in all_fitnesses if f == 999.0]
    walking = [f for f in all_fitnesses if f not in (10.0, 999.0)]

    best = min(all_fitnesses)
    worst = max(all_fitnesses)
    avg = np.mean(all_fitnesses)
    median = np.median(all_fitnesses)

    # Morphology stats
    node_counts = []
    joint_counts = []
    inherited_count = 0
    for ind in survivors:
        morph = ind.genotype.get("morph", {})
        node_counts.append(len(morph.get("nodes", {})))
        joint_counts.append(
            sum(
                1
                for n in morph.get("nodes", {}).values()
                if n.get("type") == "HINGE"
            ),
        )
        if ind.tags.get("brain") is not None:
            inherited_count += 1

    # Print summary
    console.rule(
        f"[bold cyan]Generation {GENERATION}/{BUDGET}[/bold cyan]",
    )
    console.log(
        f"  [bold]Fitness[/bold]    "
        f"best={best:.4f}  avg={avg:.4f}  "
        f"median={median:.4f}  worst={worst:.4f}",
    )
    console.log(
        f"  [bold]Status[/bold]     "
        f"walking={len(walking)}  crashed={len(crashed)}  "
        f"invalid={len(invalid)}  "
        f"total={len(survivors)}",
    )
    if walking:
        console.log(
            f"  [bold green]Walking[/bold green]    "
            f"best={min(walking):.4f}  avg={np.mean(walking):.4f}",
        )
    console.log(
        f"  [bold]Morphology[/bold] "
        f"avg_nodes={np.mean(node_counts):.1f}  "
        f"avg_hinges={np.mean(joint_counts):.1f}  "
        f"brains={inherited_count}/{len(survivors)}",
    )

    # Save videos of top 3
    sorted_survivors = sorted(
        survivors, key=lambda ind: ind.fitness_ or 999.0,
    )
    top_n = sorted_survivors[:3]
    vid_dir = DATA / "videos" / f"gen_{GENERATION:03d}"
    vid_dir.mkdir(parents=True, exist_ok=True)

    for rank, ind in enumerate(top_n):
        _save_video(ind, vid_dir, rank=rank + 1, generation=GENERATION)

    console.log(
        f"  [bold]Videos[/bold]     saved top 3 to {vid_dir}",
    )

    GENERATION += 1
    return population


def _save_video(
    ind: Individual,
    vid_dir: Path,
    rank: int,
    generation: int,
) -> None:
    """Render and save a video of an individual."""
    mujoco.set_mjcb_control(None)

    robot = genotype_to_phenotype(ind)
    if robot is None:
        return

    try:
        test_model = robot.spec.compile()
    except Exception:
        return
    if test_model.nu == 0:
        return
    del test_model

    world = SimpleFlatWorld(load_precompiled=False)
    world.spawn(
        robot.spec,
        position=SPAWN_POSITION,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Setup NN controller
    num_joints = len(data.qpos) - 7
    input_dim = 3 + num_joints + 2
    network = Network(
        input_size=input_dim,
        output_size=model.nu,
        hidden_size=HIDDEN_SIZE,
    )

    brain_weights = ind.tags.get("brain")
    if brain_weights is not None:
        expected = count_parameters(network)
        if len(brain_weights) == expected:
            fill_parameters(network, torch.tensor(brain_weights))

    # Control callback
    current_ctrl = np.zeros(model.nu)

    def ctrl_fn(m, d):
        nonlocal current_ctrl
        step = int(np.ceil(d.time / m.opt.timestep))
        if step % CONTROL_STEP_FREQ == 0:
            state = get_robot_state(d)
            phase = [
                2 * np.sin(d.time * 2.0 * np.pi),
                2 * np.cos(d.time * 2.0 * np.pi),
            ]
            full_state = np.concatenate([
                state, phase,
            ]).astype(np.float32)
            current_ctrl = network.forward(full_state)
        d.ctrl[:] = current_ctrl

    mujoco.set_mjcb_control(ctrl_fn)
    mujoco.mj_resetData(model, data)

    fitness_str = f"{ind.fitness_:.4f}" if ind.fitness_ is not None else "na"
    recorder = VideoRecorder(
        output_folder=str(vid_dir),
        file_name=f"rank{rank}_fit{fitness_str}",
    )
    video_renderer(model, data, duration=DURATION, video_recorder=recorder)
    mujoco.set_mjcb_control(None)


# ============================================================================ #
#                          INDIVIDUAL CREATION                                 #
# ============================================================================ #


def create_individual() -> Individual:
    """Create a new individual with a valid bipedal tree genome."""
    while True:
        genome = random_bipedal_tree(MAX_MODULES_PER_LEG)

        # Check morphology constraints (leg length, hinges)
        reason = validate_morphology(genome)
        if reason is not None:
            continue

        try:
            validate_genome_dict(genome.to_dict())
        except ValueError:
            continue

        ind = Individual()
        ind.genotype = {"morph": genome.to_dict()}
        ind.tags["brain"] = None
        ind.tags["requires_lr"] = True
        ind.tags["ps"] = False

        # Verify actuators exist
        if get_joint_count(ind) == 0:
            continue

        # Check it doesn't immediately crash under gravity
        spawn_reason = validate_spawned_robot(ind)
        if spawn_reason is not None:
            continue

        return ind


# ============================================================================ #
#                                MAIN                                          #
# ============================================================================ #


def main() -> None:
    console.rule(
        "[bold purple]Bipedal Walker Evolution"
        " (Tree Genome + NN + CMA-ES)[/bold purple]",
    )
    console.log(
        f"mu={MU}, lambda={LAMBDA}, generations={BUDGET}, "
        f"duration={DURATION}s, lr_budget={LEARNING_BUDGET}, "
        f"workers={NUM_WORKERS}",
    )

    # Create initial population
    console.log("Initializing population...")
    population = Population([create_individual() for _ in range(MU)])

    # Initial learning + evaluation
    population = learning(population)
    population = evaluate(population)
    population = diagnostics(population)

    # EA configuration
    config = EASettings(
        is_maximisation=False,
        num_steps=BUDGET,
        target_population_size=MU,
        output_folder=DATA,
        db_file_name="database.db",
        db_handling="delete",
    )

    # EA pipeline
    ops = [
        parent_selection(),
        reproduction(),
        learning(),
        evaluate(),
        survivor_selection(),
        diagnostics(),
    ]

    ea = EA(
        population,
        operations=ops,
        num_steps=BUDGET,
        db_file_path=config.db_file_path,
        db_handling=config.db_handling,
    )
    ea.run()

    # Get best
    best = ea.get_solution("best", only_alive=False)
    console.rule("[bold green]Best Result")
    console.log(f"Best Fitness: {best.fitness:.4f}")

    # Visualize
    if args.visualize:
        console.log("[cyan]Launching viewer for best individual...[/cyan]")
        run_simulation(best, mode="launcher")


if __name__ == "__main__":
    main()
