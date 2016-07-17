# Using .pyi files is a cool way to 'stub' out functions. This way you can specify the type hints for functions
# without having them clutter up your actual code files. PyCharm will still be able to apply the type hints to your
# uses of the functions in other code.

import typing

from compiler.compiler_state import CompilerState
from tac.tac_generation import TacInstruction
import tac.instructions as taci
import mips.configurations as config


class MipsGenerator(object):
    def __init__(self, compiler_state:CompilerState,
                 temporary_registers:typing.Tuple[str]=config.TEMPROARY_REGISTER_SET,
                 spill_mem_label:typing.Union(taci.Label, str)=config.SPILL_MEM_LABEL,
                 spill_mem_size:int=config.SPILL_MEM_SIZE, inject_source:bool=False, inject_3ac:bool=False,
                 inject_comments:bool=False): ...

    def load(self, source_tac: typing.List[TacInstruction]): ...

    def dumps(self) -> str: ...

    def dump(self, file_name: str = None): ...

    def translate_tac_to_mips(self): ...
