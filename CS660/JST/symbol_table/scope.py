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

from bintrees import RBTree
from symbol_table.symbol import Symbol


class Scope(object):
    # Static constants uses for indicating the result of the inserting
    # a symbol into the Scope.
    INSERT_SUCCESS = 0
    INSERT_SHADOWED = 1
    INSERT_REDECL = 2

    # Initializes the Scope with a balanced Red-Black tree.
    def __init__(self):
        self.map = RBTree()

    # Inserts a symbol into the Scope.
    #
    # @return INSERT_SUCCESS or INSERT_REDECL.
    def insert(self, symbol):
        if not isinstance(symbol, Symbol):
            raise TypeError("'symbol' is not an instance of Symbol.")

        if symbol.identifier not in self.map:
            self.map[symbol.identifier] = symbol
            return Scope.INSERT_SUCCESS
        else:
            return Scope.INSERT_REDECL

    # Finds a symbol in the Scope.
    #
    # @param name String identifier for the Symbol to find.
    #
    # @return Symbol if found or None if not found.
    def find(self, name):
        return self.map.get(name, None)

    # Size of the scope.
    #
    # @return Number of Symbols in the Scope.
    def size(self):
        return len(self.map)

    # Clones the current Scope and returns a deep copy.
    #
    # @return Copy of the Scope.
    def clone(self):
        result = Scope()
        for identifier, value in self.map.items():
            result.map[identifier] = value.clone()
        return result

    # @return String representation of the current Scope.
    def __repr__(self):
        symbols = []
        for key, value in self.map.items():
            symbols.append('  ' + key + ' : ' + repr(value))
        return '\n'.join(symbols)
