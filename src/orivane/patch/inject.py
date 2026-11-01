from functools import wraps

from .patch_errors import InvalidPatchMethodException
from .patch_type import PatchType

def inject(method: str, at: PatchType):
    def decorator(func):    
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, "__orivane_patched__", False):
                raise InvalidPatchMethodException(f"{func.__name__} can only be used inside a class decorated with @patch.")

            if getattr(self, "__orivane_model__", None) is None:
                raise InvalidPatchMethodException(f"{type(self).__name__} doesn't contain an __orivane_model__.")

            return func(self, *args, **kwargs)

        wrapper.__orivane_injection__ = {
            "method": method,
            "at": at,
        }

        return wrapper

    return decorator