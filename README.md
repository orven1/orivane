# Orivane

Orivane is a small Python library in alpha for making experimental changes to PyTorch models. It wraps methods with input, output, or replacement callbacks and registers patch modules with the model so they participate in device movement, training mode, checkpoints, and optimizers.

## Installation

```bash
uv add orivane
```

## Add a trainable adapter

Patch classes inherit from `PatchBase` and call `super().__init__()` so PyTorch can register modules assigned to the patch. By default, applying a patch freezes the model's existing parameters while leaving the patch's trainable parameters trainable.

```python
import torch
import torch.nn as nn

from orivane import PatchBase, PatchType, inject, patch

model = nn.Linear(10, 5)

@patch(model=model)
class AdapterPatch(PatchBase):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(5, 16),
            nn.GELU(),
            nn.Linear(16, 5),
        )

    @inject(method="forward", at=PatchType.AFTER)
    def add_adapter(self, output):
        return output + self.adapter(output)

optimizer = torch.optim.AdamW(
    (parameter for parameter in model.parameters() if parameter.requires_grad),
    lr=1e-3,
)

x = torch.randn(32, 10)
target = torch.randn(32, 5)
loss = nn.MSELoss()(model(x), target)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Restore the original method and parameter trainability when finished.
AdapterPatch.unpatch()
```

The decorated class name refers to the applied patch instance. Its added modules are registered under the model, so they are included in `model.parameters()` and `model.state_dict()`, and follow `model.to(...)` and `model.train()` calls. Patch modules are moved to the device and dtype of the model's first parameter when the patch is applied.

Use `train_only_added=False` to leave the model's existing trainability settings unchanged:

```python
@patch(model=model, train_only_added=False)
class FineTunePatch(PatchBase):
    def __init__(self):
        super().__init__()
```

Unpatching is safe to call more than once. It restores the wrapped method and the original `requires_grad` state of model parameters, and removes the patch module from the model.

## Target a nested module

`inject(method=...)` accepts a method name on the model or a dotted path to a nested module method. For example, `"layers.0.forward"` targets the first module in a `ModuleList` named `layers`:

```python
class BlockPatch(PatchBase):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    @inject(method="layers.0.forward", at=PatchType.AFTER)
    def scale_output(self, output):
        return output * (1 + self.scale)
```

Apply it with `@patch(model=model)` as in the adapter example. `BEFORE` callbacks receive the method's positional and keyword arguments and may return `(new_args, new_kwargs)`; `AFTER` callbacks receive and return the method result; `OVERWRITE` callbacks replace the method implementation.

## Patch types

| Patch type | Behavior |
|---|---|
| `PatchType.BEFORE` | Optionally changes the arguments before the original method runs. |
| `PatchType.AFTER` | Changes the original method's result. |
| `PatchType.OVERWRITE` | Replaces the original method; only one overwrite is allowed per method. |

Patches operate on PyTorch modules and methods. They do not provide model-specific integrations, automatic layer discovery, distributed-training support, or quantization support.