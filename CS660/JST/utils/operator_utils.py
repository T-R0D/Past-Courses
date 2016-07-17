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
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

def operator_to_name(operator):
        if operator == '+' or operator == '+=':
            return 'PLUS'
        elif operator == '-' or operator == '-=':
            return 'MINUS'
        elif operator == '*' or operator == '*=':
            return 'MULT'
        elif operator == '/' or operator == '/=':
            return 'DIVIDE'
        elif operator == '%' or operator == '%=':
            return 'MOD'
        elif operator == '>>' or operator == '>>=':
            return 'RSHIFT'
        elif operator == '<<' or operator == '<<=':
            return 'LSHIFT'
        elif operator == '^' or operator == '^=':
            return 'XOR'
        elif operator == '~' or operator == '~=':
            return 'BITNOT'
        elif operator == '&' or operator == '&=':
            return 'BITAND'
        elif operator == '|' or operator == '|=':
            return 'BITOR'
        elif operator == '!':
            return 'NOT'
        elif operator == '!=':
            return 'NOT_EQUAL'
        elif operator == '==':
            return 'EQUAL'
        elif operator == '&&':
            return 'AND'
        elif operator == '||':
            return 'OR'
        elif operator == '<':
            return 'LESS'
        elif operator == '<=':
            return 'LESS_EQUAL'
        elif operator == '>':
            return 'GREATER'
        elif operator == '>=':
            return 'GREATER_EQUAL'
        elif operator == '->':
            return 'ARROW'
        elif operator == '.':
            return 'DOT'
        elif operator == '++':
            return 'PLUS_PLUS'
        elif operator == '--':
            return 'MINUS_MINUS'
        elif operator == '=':
            return 'ASSIGN'
        else:
            raise Exception("Unknown operator can't be identified: {}".format(operator))