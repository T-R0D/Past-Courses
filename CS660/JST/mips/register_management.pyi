# Using .pyi files is a cool way to 'stub' out functions. This way you can specify the type hints for functions
# without having them clutter up your actual code files. PyCharm will still be able to apply the type hints to your
# uses of the functions in other code.

import typing


class OutOfSpillMemoryException(Exception):
    pass


class MipsRegisterUseTable(object):
    def __init__(self, available_registers: typing.Tuple[str], spill_memory_base_label: str, spill_memory_size: int,
                 word_size: int = 4): ...

    def acquire(self, pseudo_register: str) -> dict: ...

    def release(self, pseudo_register: str): ...

    def release_all(self): ...

    def _spill(self, pseudo_register: str, physical_register: str): ...

    def _recover(self, pseudo_register: str) -> typing.Tuple[str, list]: ...

    def __str__(self): ...

    def __repr__(self): ...
