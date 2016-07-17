# This file is part of JST.
#
# JST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#  
# JST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#  
# You should have received a copy of the GNU General Public License
# along with JST.  If not, see <http://www.gnu.org/licenses/>.

import mips.configurations as config
import mips.registers as mr
import mips.instructions as mi


class Macro(object):
    def __init__(self, name, args=None, body=None):
        self.name = name
        self.args = args if args else []
        self.body = body if body else []

    def definition(self):
        return [mi.BEGIN_MACRO(self.name, self.args)] + self.body + [mi.END_MACRO()]

    def call(self, *args):
        return mi.CALL_MACRO(self.name, args)


__T_REGISTERS = (mr.T0, mr.T1, mr.T2, mr.T3, mr.T4, mr.T5, mr.T6, mr.T7, mr.T8, mr.T9)


__save_register_macro_instructions = [mi.COMMENT("brace yourself for a long, unrolled loop...")]
for __temp_register in __T_REGISTERS:
    __save_register_macro_instructions.append(mi.SW(__temp_register, mi.offset_from_register_with_immediate(mr.SP)))
    __save_register_macro_instructions.append(mi.SUBIU(mr.SP, mr.SP, mr.WORD_SIZE))
if config.NOT_TESTING_FUNCTIONS:
    __save_register_macro_instructions = []
SAVE_REGISTER_MACRO = Macro(name='SAVE_T_REGISTERS', args=None, body=__save_register_macro_instructions)


__restore_register_macro_instructions = [mi.COMMENT("brace yourself for a long, unrolled loop...")]
for __temp_register in reversed(__T_REGISTERS):
    __restore_register_macro_instructions.append(mi.ADDIU(mr.SP, mr.SP, mr.WORD_SIZE))
    __restore_register_macro_instructions.append(mi.LW(__temp_register, mi.offset_from_register_with_immediate(mr.SP)))
if config.NOT_TESTING_FUNCTIONS:
    __restore_register_macro_instructions = []
RESTORE_REGISTER_MACRO = Macro(name='RESTORE_T_REGISTERS', args=None, body=__restore_register_macro_instructions)



__save_spill_mem_macro_body = [mi.COMMENT("brace yourself for a long, unrolled loop...")]
for i in range(0, config.SPILL_MEM_SIZE, mr.WORD_SIZE):
    __save_spill_mem_macro_body.extend([
        mi.LW(mr.A3, mi.offset_label_immediate('SPILL_MEMORY', i)),
        mi.SW(mr.A3, mi.offset_from_register_with_immediate(mr.SP)),
        mi.SUBIU(mr.SP, mr.SP, mr.WORD_SIZE)
    ])
if config.NOT_TESTING_FUNCTIONS:
    __save_spill_mem_macro_body = []
SAVE_SPILL_MEM_MACRO = Macro(name='SAVE_SPILL_MEM', args=None, body=__save_spill_mem_macro_body)


__restore_spill_mem_macro_body = [mi.COMMENT("brace yourself for a long, unrolled loop...")]
for i in range(config.SPILL_MEM_SIZE - mr.WORD_SIZE, -mr.WORD_SIZE, -mr.WORD_SIZE):
    __restore_spill_mem_macro_body.extend([
        mi.ADDIU(mr.SP, mr.SP, mr.WORD_SIZE),
        mi.LW(mr.A3, mi.offset_from_register_with_immediate(mr.SP)),
        mi.SW(mr.A3, mi.offset_label_immediate('SPILL_MEMORY', i))
    ])
if config.NOT_TESTING_FUNCTIONS:
    __restore_spill_mem_macro_body = []
RESTORE_SPILL_MEM_MACRO = Macro(name='RESTORE_SPILL_MEM', args=None, body=__restore_spill_mem_macro_body)


__caller_function_prologue_body = [
    mi.COMMENT("caller should save it's own $ra, $fp, and registers"),
    mi.SW(mr.RA, mi.offset_from_register_with_immediate(mr.SP)),
    mi.SUBIU(mr.SP, mr.SP, mr.WORD_SIZE),
    mi.SW(mr.FP, mi.offset_from_register_with_immediate(mr.SP)),
    mi.SUBIU(mr.SP, mr.SP, mr.WORD_SIZE),
    mi.COMMENT("caller pushes registers and spill memory onto the stack as well"),
    SAVE_REGISTER_MACRO.call(),
    SAVE_SPILL_MEM_MACRO.call(),
    mi.COMMENT("save the value of $sp here into $a0 as temporary storage until the arguments are moved"),
    mi.COMMENT("$fp needs to stay where it's at while the arguments are copied after this macro"),
    mi.ADD(mr.A0, mr.SP, mr.ZERO)
]
CALLER_FUNCTION_PROLOGUE_MACRO = Macro(name='CALLER_FUNCTION_PROLOGUE', args=None, body=__caller_function_prologue_body)


__callee_function_prologue_body = [
    mi.COMMENT("set $fp to the proper spot by recovering the value from $a0"),
    mi.ADD(mr.FP, mr.A0, mr.ZERO),
    mi.COMMENT("allocate stack space for variables ($sp = $sp - space for variables)"),
    mi.LI(mr.A0, mr.WORD_SIZE),
    mi.MULU(mr.A1, mr.A0, mi.macro_arg('variable_size')),
    mi.SUB(mr.SP, mr.SP, mr.A1)
]
CALLEE_FUNCTION_PROLOGUE_MACRO = Macro(name='CALLEE_FUNCTION_PROLOGUE',
                                       args=['variable_size'],
                                       body=__callee_function_prologue_body)


__callee_function_epilogue_body = [
    mi.COMMENT("de-allocate the memory used for local variables and parameters"),
    mi.ADD(mr.SP, mr.FP, mr.ZERO),
    mi.COMMENT("jump back to the caller"),
    mi.JR(mr.RA)
]
CALLEE_FUNCTION_EPILOGUE_MACRO = Macro(name='CALLEE_FUNCTION_EPILOGUE',
                                       args=None,
                                       body=__callee_function_epilogue_body)

__caller_function_epilogue_body = [
    mi.COMMENT("recover the spill memory and the stored registers"),
    RESTORE_SPILL_MEM_MACRO.call(),
    RESTORE_REGISTER_MACRO.call(),
    mi.COMMENT("recover the caller's $fp and $ra"),
    mi.ADDIU(mr.SP, mr.SP, mr.WORD_SIZE),
    mi.LW(mr.FP, mi.offset_from_register_with_immediate(mr.SP)),
    mi.ADDIU(mr.SP, mr.SP, mr.WORD_SIZE),
    mi.LW(mr.RA, mi.offset_from_register_with_immediate(mr.SP)),
]
CALLER_FUNCTION_EPILOGUE_MACRO = Macro(name='CALLER_FUNCTION_EPILOGUE',
                                       args=None,
                                       body=__caller_function_epilogue_body)


__logical_and_false_label = '__LAND_FALSE'
__logical_and_end_label = '__LAND_END'
__logical_and_body = [
    mi.BEQZ(mi.macro_arg('lhs'), __logical_and_false_label),
    mi.BEQZ(mi.macro_arg('rhs'), __logical_and_false_label),
    mi.LI(mr.A2, 1),
    mi.J(__logical_and_end_label),
    mi.LABEL(__logical_and_false_label),
    mi.LI(mr.A2, 0),
    mi.LABEL(__logical_and_end_label)
]
LAND_MACRO = Macro(name='__LAND', args=['lhs', 'rhs'], body=__logical_and_body)


__logical_or_true_label = '__LOR_TRUE'
__logical_or_end_label = '__LOR_END'
__logical_and_body = [
    mi.BEQZ(mi.macro_arg('lhs'), __logical_or_true_label),
    mi.BEQZ(mi.macro_arg('rhs'), __logical_or_true_label),
    mi.LI(mr.A2, 0),
    mi.J(__logical_or_end_label),
    mi.LABEL(__logical_or_true_label),
    mi.LI(mr.A2, 1),
    mi.LABEL(__logical_or_end_label)
]
LOR_MACRO = Macro(name='__LOR', args=['lhs', 'rhs'], body=__logical_and_body)
