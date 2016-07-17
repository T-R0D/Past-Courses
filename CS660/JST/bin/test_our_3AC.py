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
from compiler.compiler_state import CompilerState


"""
USAGE:

To try our thing, write your own C program in the string below, then click the green arrow in the upper
right. You should see an output window appear on the bottom. That terminal shows all output, even warnings that we
do not generate, so please ignore them. Enjoy.
"""


def main():
    ##
    #  WRITE YOUR C PROGRAM IN THIS STRING!
    ##
    your_c_program = """
      // your program here!
      int main() {
        int my_3d_array[1][2][3][4];
        return 0;
      }
    """

    compiler_state = CompilerState()
    ast = compiler_state.parse(your_c_program)

    ast.to_3ac(include_source=True)

if __name__ == '__main__':
    main()