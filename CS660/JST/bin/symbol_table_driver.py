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

###############################################################################
# File Description: External driver for symbol table test.
###############################################################################

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from symbol_table.symbol_table import SymbolTable
from symbol_table.scope import Scope
from symbol_table.symbol import Symbol


def main():
    sym = SymbolTable()

    sym.insert(Symbol('Apple'))
    sym.insert(Symbol('Banana'))
    sym.push()
    sym.insert(Symbol('Cantaloupe'))
    print(sym)

    assert sym.insert(Symbol('Apple')) is Scope.INSERT_SHADOWED
    assert sym.insert(Symbol('Cantaloupe')) is Scope.INSERT_REDECL
    assert sym.insert(Symbol('Blueberries')) is Scope.INSERT_SUCCESS

    assert sym.size() is 2

    found = sym.find('Apple')
    assert found is not None
    assert found[0].identifier is 'Apple'

    found = sym.find('Cantaloupe')
    assert found is not None
    assert found[0].identifier is 'Cantaloupe'

    found = sym.find('Durian')
    assert found is None

    sym.pop()
    assert sym.size() is 1

if __name__ == '__main__':
    main()
