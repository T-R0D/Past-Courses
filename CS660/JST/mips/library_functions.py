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

import mips.instructions as mi
import mips.registers as mr
import mips.macros as mm
import symbol_table.symbol as symbol


class FunctionDefinition(object):
    def __init__(self, identifier, body):
        self.identifier = identifier
        self.body = body

    def get_mips(self):
        return [mi.LABEL(self.identifier)] + self.body


#
# PRINT CHAR
#
__PRINT_CHAR_SYSCALL = 11

__print_char_return_type = symbol.TypeDeclaration()
__print_char_return_type.add_type_specifier('int')

__print_char_parameter = symbol.VariableSymbol('x', 0, 0)
__print_char_parameter_type = symbol.TypeDeclaration()
__print_char_parameter_type.add_type_specifier('int')
__print_char_parameter.add_type_declaration(__print_char_parameter_type)

PrintCharDeclaration = symbol.FunctionSymbol('print_char', 0, 0)
PrintCharDeclaration.set_named_parameters([__print_char_parameter])
PrintCharDeclaration.add_return_type_declaration(__print_char_return_type)

__print_char_body = [
    mm.CALLEE_FUNCTION_PROLOGUE_MACRO.call('0'),
    mi.COMMENT("load $v0 with the value for the print char syscall"),
    mi.LI(mr.V0, __PRINT_CHAR_SYSCALL),
    mi.COMMENT("the first (and only) argument is the value to print"),
    mi.LW(mr.A0, mi.offset_from_register_with_immediate(mr.FP)),
    mi.SYSCALL(),
    mm.CALLEE_FUNCTION_EPILOGUE_MACRO.call()
]
PrintCharDefinition = FunctionDefinition(identifier='print_char', body=__print_char_body)


#
# PRINT INTEGER
#
__PRINT_INT_SYSCALL = 1

__print_int_return_type = symbol.TypeDeclaration()
__print_int_return_type.add_type_specifier('int')

__print_int_parameter = symbol.VariableSymbol('x', 0, 0)
__print_int_parameter_type = symbol.TypeDeclaration()
__print_int_parameter_type.add_type_specifier('int')
__print_int_parameter.add_type_declaration(__print_int_parameter_type)

PrintIntDeclaration = symbol.FunctionSymbol('print_int', 0, 0)
PrintIntDeclaration.set_named_parameters([__print_int_parameter])
PrintIntDeclaration.add_return_type_declaration(__print_int_return_type)

__print_int_body = [
    mm.CALLEE_FUNCTION_PROLOGUE_MACRO.call('0'),
    mi.COMMENT("load $v0 with the value for the print int syscall"),
    mi.LI(mr.V0, __PRINT_INT_SYSCALL),
    mi.COMMENT("the first (and only) argument is the value to print"),
    mi.LW(mr.A0, mi.offset_from_register_with_immediate(mr.FP)),
    mi.SYSCALL(),
    mm.CALLEE_FUNCTION_EPILOGUE_MACRO.call()
]
PrintIntDefinition = FunctionDefinition(identifier='print_int', body=__print_int_body)


#
# PRINT STRING
#
__PRINT_STRING_SYSCALL = 4

__print_string_return_type = symbol.TypeDeclaration()
__print_string_return_type.add_type_specifier('int')

__print_string_parameter = symbol.VariableSymbol('s', 0, 0)
__print_string_parameter_type = symbol.TypeDeclaration()
__print_string_parameter_type.add_type_specifier('char')
__print_string_parameter.add_type_declaration(__print_string_parameter_type)
__print_string_parameter.array_dims = [symbol.VariableSymbol.EMPTY_ARRAY_DIM]

PrintStringDeclaration = symbol.FunctionSymbol('print_string', 0, 0)
PrintStringDeclaration.set_named_parameters([__print_string_parameter])
PrintStringDeclaration.add_return_type_declaration(__print_string_return_type)

__print_string_body = [
    mm.CALLEE_FUNCTION_PROLOGUE_MACRO.call('0'),
    mi.COMMENT("load $v0 with the value for the print string syscall"),
    mi.LI(mr.V0, __PRINT_STRING_SYSCALL),
    mi.COMMENT("the first (and only) argument is the base address of the null terminated ascii string"),
    mi.LA(mr.A0, mi.offset_from_register_with_immediate(mr.FP)),
    mi.SYSCALL(),
    mm.CALLEE_FUNCTION_EPILOGUE_MACRO.call()
]
PrintStringDefinition = FunctionDefinition(identifier='print_string', body=__print_string_body)


#
# PRINT FLOAT
#
__PRINT_FLOAT_SYSCALL = 2

__print_float_return_type = symbol.TypeDeclaration()
__print_float_return_type.add_type_specifier('int')

__print_float_parameter = symbol.VariableSymbol('x', 0, 0)
__print_float_parameter_type = symbol.TypeDeclaration()
__print_float_parameter_type.add_type_specifier('float')
__print_float_parameter.add_type_declaration(__print_float_parameter_type)

PrintFloatDeclaration = symbol.FunctionSymbol('print_float', 0, 0)
PrintFloatDeclaration.set_named_parameters([__print_float_parameter])
PrintFloatDeclaration.add_return_type_declaration(__print_float_return_type)

__print_float_body = [
    mm.CALLEE_FUNCTION_PROLOGUE_MACRO.call('0'),
    mi.COMMENT("load $v0 with the value for the print float syscall"),
    mi.LI(mr.V0, __PRINT_FLOAT_SYSCALL),
    mi.COMMENT("the first (and only) argument is the base address of the null terminated ascii string"),
    mi.LWC1(mr.F12, mi.offset_from_register_with_immediate(mr.FP)),
    mi.SYSCALL(),
    mm.CALLEE_FUNCTION_EPILOGUE_MACRO.call()
]
PrintFloatDefinition = FunctionDefinition(identifier='print_float', body=__print_float_body)
