from functools import wraps

from torch import nn

from .patch_base import PatchBase
from .patch_errors import InvalidModuleException, InvalidPatchMethodException
from .patch_type import PatchType

def _resolve_method(model: nn.Module, path: str):
    module_path, separator, method_name = path.rpartition(".")
    target = model.get_submodule(module_path) if separator else model
    method = getattr(target, method_name, None)

    if not callable(method):
        raise InvalidPatchMethodException(f"Method {path!r} does not exist on {type(model).__name__}.")

    return target, method_name, method


def _create_wrapper(original, injections):
    before = [
        callback for at, callback in injections
        if at is PatchType.BEFORE
    ]
    after = [
        callback for at, callback in injections
        if at is PatchType.AFTER
    ]
    overwrite = [
        callback for at, callback in injections
        if at is PatchType.OVERWRITE
    ]

    if len(overwrite) > 1:
        raise InvalidPatchMethodException("Only one OVERWRITE injection is allowed per method.")

    implementation = overwrite[0] if overwrite else original

    @wraps(original)
    def wrapper(*args, **kwargs):
        for callback in before:
            result = callback(*args, **kwargs)

            if result is not None:
                if (
                    not isinstance(result, tuple)
                    or len(result) != 2
                    or not isinstance(result[0], tuple)
                    or not isinstance(result[1], dict)
                ):
                    raise TypeError("BEFORE injections must return None or (args: tuple, kwargs: dict).")

                args, kwargs = result

        output = implementation(*args, **kwargs)

        for callback in after:
            output = callback(output)

        return output

    return wrapper


def _next_patch_name(model: nn.Module) -> str:
    index = 0

    while f"_orivane_patch_{index}" in model._modules:
        index += 1

    return f"_orivane_patch_{index}"


def patch(*, model: nn.Module, train_only_added: bool = True):
    if not isinstance(model, nn.Module):
        raise InvalidModuleException("'model' must inherit from torch.nn.Module.")

    def decorator(cls):
        if not issubclass(cls, PatchBase):
            raise InvalidPatchMethodException(f"{cls.__name__} must inherit from PatchBase.")

        cls.__orivane_patch_decorator_applied__ = True
        instance = cls()
        instance._set_model(model)

        reference = next(model.parameters(), None)
        if reference is not None:
            instance.to(device=reference.device, dtype=reference.dtype)

        injections = {}
        for name, member in vars(cls).items():
            metadata = getattr(member, "__orivane_injection__", None)

            if metadata is None:
                continue

            method_path = metadata["method"]
            at = metadata["at"]

            if not isinstance(method_path, str) or not method_path:
                raise InvalidPatchMethodException("Injected method paths must be non-empty strings.")
            
            if not isinstance(at, PatchType):
                raise InvalidPatchMethodException(f"Invalid patch type for method {method_path!r}.")

            injections.setdefault(method_path, []).append((at, getattr(instance, name)))

        restorations = []
        for method_path, callbacks in injections.items():
            target, method_name, original = _resolve_method(model, method_path)
            had_instance_attr = method_name in target.__dict__
            old_instance_attr = target.__dict__.get(method_name)
            restorations.append(
                (
                    target,
                    method_name,
                    had_instance_attr,
                    old_instance_attr,
                    _create_wrapper(original, callbacks),
                )
            )

        patch_name = _next_patch_name(model)
        instance.__orivane_patched__ = True
        original_trainability = []
        applied_restorations = []
        registered = False

        try:
            model.add_module(patch_name, instance)
            registered = True

            if train_only_added:
                patch_parameter_ids = {
                    id(parameter) for parameter in instance.parameters()
                }
                for parameter in model.parameters():
                    if id(parameter) not in patch_parameter_ids:
                        original_trainability.append((parameter, parameter.requires_grad))
                        parameter.requires_grad_(False)

            for target, method_name, existed, old_value, wrapped in restorations:
                setattr(target, method_name, wrapped)
                applied_restorations.append((target, method_name, existed, old_value))

        except Exception:
            for target, method_name, existed, old_value in reversed(applied_restorations):
                if existed:
                    setattr(target, method_name, old_value)
                else:
                    delattr(target, method_name)

            for parameter, requires_grad in original_trainability:
                parameter.requires_grad_(requires_grad)

            if registered:
                delattr(model, patch_name)

            instance.__orivane_patched__ = False
            
            raise

        active = True

        def unpatch():
            nonlocal active
            if not active:
                return

            for target, method_name, existed, old_value in reversed(applied_restorations):
                if existed:
                    setattr(target, method_name, old_value)
                else:
                    delattr(target, method_name)

            for parameter, requires_grad in original_trainability:
                parameter.requires_grad_(requires_grad)

            if model._modules.get(patch_name) is instance:
                delattr(model, patch_name)

            instance.__orivane_patched__ = False
            active = False

        instance.unpatch = unpatch

        return instance

    return decorator
