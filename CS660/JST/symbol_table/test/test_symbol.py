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

import unittest

from symbol_table.symbol import Symbol, FunctionSymbol, TypeDeclaration, VariableSymbol


class TestSymbol(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_variable_str(self):
        type = TypeDeclaration()
        type.add_type_specifier('int')
        symbol = VariableSymbol(identifier='my_variable', lineno=0, column=0)
        symbol.add_type_declaration(type)

        s = str(symbol)

        self.assertEqual("int my_variable", s)

    def test_function_str(self):
        # TODO: with the introduction of the FunctionSymbol, we might be throwing out this test, so it's ok if it fails
        type = TypeDeclaration()
        type.add_type_specifier('int')
        symbol = FunctionSymbol(identifier='my_function', lineno=0, column=0)
        symbol.add_return_type_declaration(type)

        parameter_1 = VariableSymbol(identifier='parameter_1', lineno=0, column=0)
        p_1_type = TypeDeclaration()
        p_1_type.add_type_specifier('int')
        parameter_1.add_type_declaration(p_1_type)

        parameter_2 = VariableSymbol(identifier='parameter_2', lineno=0, column=0)
        p_2_type = TypeDeclaration()
        p_2_type.add_type_specifier('char')
        parameter_2.add_type_declaration(p_2_type)


        parameters = [parameter_1, parameter_2]
        symbol.set_named_parameters(parameters)

        s = str(symbol)
        self.assertEqual("int my_function(int parameter_1, char parameter_2)", s)

    def test_size_in_bytes(self):
        symbol = VariableSymbol('ignored', 0, 0)
        symbol.type_specifiers = 'char'
        self.assertEqual(1, symbol.size_in_bytes())

        symbol = VariableSymbol('ignored', 0, 0)
        symbol.type_specifiers = 'int'
        self.assertEqual(4, symbol.size_in_bytes())

        symbol = VariableSymbol('ignored', 0, 0)
        symbol.type_specifiers = 'float'
        self.assertEqual(4, symbol.size_in_bytes())

        # TODO: add these back in if we implement pointers
        # symbol = VariableSymbol('ignored', 0, 0)
        # symbol.type_specifiers = 'char'
        # symbol.pointer_modifiers = [PointerDeclaration()]
        # self.assertEqual(4, symbol.size_in_bytes())
        #
        # symbol = VariableSymbol('ignored', 0, 0)
        # symbol.type_specifiers = 'char'
        # symbol.pointer_modifiers = [PointerDeclaration(), PointerDeclaration()]
        # self.assertEqual(4, symbol.size_in_bytes())

        symbol = VariableSymbol('ignored', 0, 0)
        symbol.type_specifiers = 'char'
        symbol.array_dims = [4]
        self.assertEqual(4, symbol.size_in_bytes())

        symbol = VariableSymbol('ignored', 0, 0)
        symbol.type_specifiers = 'char'
        symbol.array_dims = [2, 2]
        self.assertEqual(4, symbol.size_in_bytes())

        # symbol = VariableSymbol('ignored', 0, 0)
        # symbol.type_specifiers = 'char'
        # symbol.pointer_modifiers = [PointerDeclaration(), PointerDeclaration()]
        # symbol.array_dims = [4]
        # self.assertEqual(16, symbol.size_in_bytes())
