"""Small runtime corrections for the pinned upstream training loop."""
import ast
from pathlib import Path


def compile_training_program(path):
    path = Path(path)
    tree = ast.parse(path.read_text(), filename=str(path))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loops = [node for node in main.body if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "epoch"]
    if len(loops) != 1:
        raise RuntimeError("Pinned upstream epoch loop changed; review the stopping guard")
    # The upstream inner break alone can allow resumed runs to exceed max_train_steps.
    guard = ast.parse("if global_step >= args.max_train_steps:\n    break").body[0]
    ast.copy_location(guard, loops[0])
    loops[0].body.insert(0, guard)
    return compile(ast.fix_missing_locations(tree), str(path), "exec")


def save_rank_random_state(accelerator, output_dir):
    from accelerate.checkpointing import save_accelerator_state
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Accelerate's own format restores Python, NumPy, Torch, CUDA and accumulation step.
    save_accelerator_state(output_dir, [], [], [], [], accelerator.process_index, accelerator.step)


def save_training_generators(output_dir, rank, scope):
    import torch
    states = {name: (generator.get_state() if isinstance(generator, torch.Generator)
                     else generator.bit_generator.state)
              for name in ["rng", "torch_rng", "index_rng"]
              if (generator := scope.get(name)) is not None}
    torch.save(states, Path(output_dir) / f"training_random_states_{rank}.pkl")


def load_training_generators(input_dir, rank, scope):
    import torch
    path = Path(input_dir) / f"training_random_states_{rank}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint lacks custom training generator states: {path}")
    states = torch.load(path, map_location="cpu", weights_only=False)
    for name, state in states.items():
        generator = scope[name]
        if isinstance(generator, torch.Generator):
            generator.set_state(state)
        else:
            generator.bit_generator.state = state
