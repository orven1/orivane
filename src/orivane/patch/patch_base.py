from torch import nn

from .patch_errors import InvalidModuleException, InvalidPatchMethodException

class _PatchMeta(type(nn.Module)):
    def __call__(cls, *args, **kwargs):
        has_injections = any(
            getattr(member, "__orivane_injection__", None) is not None
            for member in vars(cls).values()
        )
        decorator_applied = cls.__dict__.get("__orivane_patch_decorator_applied__", False)

        if has_injections and not decorator_applied:
            raise InvalidPatchMethodException(f"{cls.__name__} contains injected methods and must be decorated with @patch.")

        return super().__call__(*args, **kwargs)


class PatchBase(nn.Module, metaclass=_PatchMeta):
    def __init__(self):
        super().__init__()
        object.__setattr__(self, "__orivane_model__", None)
        self.__orivane_added_modules__ = {}
        self.__orivane_patched__ = False

    def _set_model(self, model: nn.Module):
        object.__setattr__(self, "__orivane_model__", model)

    def add_module(self, name: str, module: nn.Module, *, trainable: bool = True, device=None,):
        model = self.__orivane_model__

        if not isinstance(module, nn.Module):
            raise InvalidModuleException("'module' has to inherit from torch.nn.Module.")

        if name in self._modules or hasattr(self, name):
            raise ValueError(f"Patch attribute {name!r} already exists.")

        if device is None and model is not None:
            reference = next(model.parameters(), None)
            if reference is not None:
                device = reference.device

        if device is not None:
            module.to(device)

        module.requires_grad_(trainable)
        super().add_module(name, module)
        self.__orivane_added_modules__[name] = module
        return module

    def trainable_parameters(self):
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )
