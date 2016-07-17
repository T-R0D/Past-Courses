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

import math

from symbol_table.scope import Scope
from symbol_table.symbol import Symbol, VariableSymbol

# This is where MARS starts the global data memory
# We may need to watch out for memory things like .asciiz grab
# So perhaps we should start somewhere else?
# (Shubham) If we allocate everything manually everything, we can ignore the .asciiz grab
MIPS_DATA_MEMORY_BASE = 0x10010000


class SymbolTable(object):

    # Initializes the symbol table
    def __init__(self):
        self.table = []

        self.next_data_memory_location = MIPS_DATA_MEMORY_BASE
        # Increments by the type size
        self.next_activation_frame_offset = 0

    # Pushes a scope onto the table.
    #
    # @param scope Scope to push. Default pushes an empty Scope.
    def push(self, scope=None):
        if scope is None:
            scope = Scope()
        if type(scope) is Scope:
            self.table.append(scope)
        else:
            raise TypeError("'scope' is not an instance of Scope.")

    # Pops the top-most Scope from the table and returns it.
    def pop(self):
        # The second scope is first scope of a function
        # if len(self.table) == 2:
        #     self.next_activation_frame_offset = 0

        return self.table.pop()

    # Inserts a symbol into the top-most Scope.
    #
    # @param symbol Symbol to insert.
    #
    # @return Tuple of the result and a list of shadowed Symbols or re-declared Symbol.
    def insert(self, symbol):
        if not isinstance(symbol, Symbol):
            raise TypeError("parameter 'symbol' is not an instance of Symbol. Actual type: {}".format(type(symbol)))
        elif not self.table:
            raise Exception('Table has no scopes available to insert into. Offending symbol: {}'.format(symbol))

        if isinstance(symbol, VariableSymbol):
            if symbol.is_array:
                if symbol.is_parameter:
                    # 1 word for the base address
                    # 1 word for total array size in bytes
                    # x words for the bound on each dimension
                    required_byte_size = 4 * (len(symbol.array_dims) + 2)
                else:
                    required_byte_size = math.ceil((symbol.size_in_bytes() * symbol.array_size) / 4) * 4
            else:
                required_byte_size = math.ceil(symbol.size_in_bytes() / 4) * 4

            if len(self.table) == 1:
                # symbol.global_memory_location = self.next_data_memory_location
                # self.next_data_memory_location += required_byte_size

                symbol.global_memory_location = symbol.identifier

            else:
                symbol.activation_frame_offset = self.next_activation_frame_offset
                self.next_activation_frame_offset += required_byte_size

        shadowed_symbols = []
        for scope in self.table:
            result = scope.find(symbol.identifier)
            if result is not None:
                if scope is not self.table[-1]:
                    shadowed_symbols.append(result)
                else:
                    return Scope.INSERT_REDECL, [result]

        self.table[-1].insert(symbol)

        if len(shadowed_symbols) is 0:
            return Scope.INSERT_SUCCESS, []
        else:
            return Scope.INSERT_SHADOWED, shadowed_symbols

    # Finds a symbol in the table by searching the top-most Scope to the
    # bottom-most Scope.
    #
    # @param name String identifier for the Symbol to find.
    #
    # @return Tuple of the Symbol and the Scope index it was found in.
    def find(self, name):
        for index, scope in enumerate(reversed(self.table)):
            result = scope.find(name)
            if result is not None:
                scope_level = (len(self.table) - (index + 1))
                return result, scope_level
        return None, None

    # Finds a symbol in the table by searching only the top-most Scope.
    #
    # @param name String identifier for the Symbol to find.
    #
    # @return 'None' or the Symbol.
    def find_in_top(self, name):
        return self.table[-1].find(name)

    # Finds a type in the table by searching the top-most Scope to the
    # bottom-most Scope.
    #
    # @param name String identifier for the type to find.
    #
    # @return Tuple of the type and the Scope index it was found in.
    def find_type(self, identifier):
        pass

    # Size of the table.
    #
    # @return Number of scopes in the table.
    def size(self):
        return len(self.table)

    # Clones the current SymbolTable and returns a deep copy.
    #
    # @return Copy of the SymbolTable.
    def clone(self):
        result = SymbolTable()
        result.table = []
        for scope in self.table:
            result.table.append(scope.clone())
        return result

    # @return String representation of the current SymbolTable.
    def __repr__(self):
        scopes = []
        for index, scope in enumerate(self.table):
            scopes.append('Scope #' + repr(index) + '\n' + repr(scope))
        return '\n'.join(scopes)
