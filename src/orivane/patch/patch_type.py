from enum import Enum

class PatchType(Enum):
    BEFORE = 0 # Changes the input for the original function, it's output is the new input for the orignal function
    AFTER = 1 # Changes the output, it's input is the output of the original function
    OVERWRITE = 2 # Overwrites the whole function