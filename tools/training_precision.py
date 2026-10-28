"""Keep optimizer updates in FP32 while Accelerate autocasts model computation."""
import torch


def prepare_full_precision_parameters(model, optimizer):
    before = {id(p) for p in model.parameters()}
    assert all(p.requires_grad for p in model.parameters()), "Expected full Transformer training"
    assert not optimizer.state, "Convert precision before optimizer state initialization"
    model.to(dtype=torch.float32)
    assert before == {id(p) for p in model.parameters()}, "Parameter identity changed"
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert before == {id(p) for p in optimized} and len(before) == len(optimized)
    assert all(p.dtype == torch.float32 for p in optimized)
    return {"parameter_storage": "float32", "optimizer_state": "float32", "compute": "bf16_autocast",
            "parameter_bytes": sum(p.numel() * p.element_size() for p in optimized)}
