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

import mips.generation as generation
from compiler.compiler_state import CompilerState
from loggers.logger import Logger


class TestMipsGenerator(unittest.TestCase):
    def setUp(self):
        self.compiler_state = CompilerState(print_productions=False)
        self.enable_debug(False)

        self.generator = generation.MipsGenerator(self.compiler_state, inject_source = True, inject_3ac=True)

    def tearDown(self):
        self.compiler_state.teardown()
        self.compiler_state = None

    def enable_debug(self, enable, productions=True, source=False):
        if enable:
            prod_logger = self.compiler_state.get_parser_logger()

            prod_logger.add_switch(Logger.INFO)
            if productions:
                prod_logger.add_switch(Logger.PRODUCTION)

            if source:
                prod_logger.add_switch(Logger.SOURCE)

    def test_plain_main(self):
        data = """
            int main()
            {
                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()

        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_global_variable_declaration(self):
        data = """
            int g = 5;

            int main() {
              return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_print_int(self):
        data = """
            int main() {
              print_int(777);
              return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_declarations(self):
        data = """
            int main()
            {
                char c;
                short s;
                int i;
                long long l;

                char arr[13];

                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_simple_assignment(self):
        data = """
            int main()
            {
                int i;
                i = 0;
                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_constant_expression(self):
        data = """
            int main()
            {
                int i, a, b;
                i = a + b;
                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_explicit_cast(self):
        # self.enable_debug(True)
        data = """
            int main()
            {
                int i;
                i = (int) 4.2;
                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_if_else(self):
        data = """
            int main()
            {
                int i;
                if (i == 5)
                {
                    i = 6;
                }
                else
                {
                    i = 5;
                }
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_if_elif_else(self):
        data = """
            int main()
            {
                int i;
                if (i == 1)
                {
                    i = 7;
                }
                else if(i == 2)
                {
                    i = 8;
                }
                else
                {
                    i = 9;
                }
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_while_loop(self):
        data = """
            int main()
            {
                int i;
                while(i < 5){}
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_array_access(self):
        data = """
            int main()
            {
                int a[2][2];
                a[0][0] = a[1][1];
                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_constant_folding(self):
        data = """
            const int a = 1 + 1;
            const int b = a + 2;
            int main()
            {
                int c = a + b;
                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_array_initializers(self):
        data = """
            int main()
            {
                int a[5] = {1,2,3,4,5};
                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_function_with_param(self):
        data = """
            int foo( int a, int b, int c);

            int main()
            {
                int i = foo(1,2,3);
                return 0;
            }

            int foo( int a, int b, int c)
            {
                return 1;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_for_loop(self):
        data = """
            int main()
            {
                int i = 0;
                int j;
                for(i=0;i<3;i++)
                {
                    j = 5;
                }
                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_do_while_loop(self):
        data = """
            int main()
            {
                int i;
                do
                {
                    i = 5;
                }
                while (i > 10);
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_matrix_multiplication(self):
        test_program = """
            const int ARRAY_DIM = 2;

            // hard code dimensions for simplicity
            int matrix_multiply(int C[ARRAY_DIM][ARRAY_DIM], int A[ARRAY_DIM][ARRAY_DIM], int B[ARRAY_DIM][ARRAY_DIM]);

            int main() {
              int i, j;
              int A[ARRAY_DIM][ARRAY_DIM], B[ARRAY_DIM][ARRAY_DIM], C[ARRAY_DIM][ARRAY_DIM];

              for (i = 0; i < ARRAY_DIM; i++) {
                for (j = 0; j < ARRAY_DIM; j++) {
                  A[i][j] = B[i][j] = 1;
                }
              }

              matrix_multiply(C, A, B);

              return 0;
            }

            int matrix_multiply(int C[ARRAY_DIM][ARRAY_DIM], int A[ARRAY_DIM][ARRAY_DIM], int B[ARRAY_DIM][ARRAY_DIM]) {
              int i, j, k;

              for (i = 0; i < ARRAY_DIM; i++) {
                for (j = 0; j < ARRAY_DIM; j++) {
                  C[i][j] = 0;
                  for (k = 0; k < ARRAY_DIM; k++) {
                    C[i][j] += A[i][j + k] * B[i + k][j];
                  }
                }
              }
            }
            """
        ast = self.compiler_state.parse(test_program)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

    def test_bubble_sort(self):
        test_program = """
            const int N_ITEMS = 5;
            int bubble_sort(int items[N_ITEMS], int n_items);

            int main() {
              int items[N_ITEMS] = {5, 1, 4, 3, 2};

              bubble_sort(items, 5);

              return 0;
            }

            int bubble_sort(int items[N_ITEMS], int n_items) {
              int i, j;
              int temp;

              for (i = 0; i < n_items; i++) {
                for (j = i; j < n_items; j++) {
                  if (items[i] < items[j]) {
                    temp = items[i];
                    items[i] = items[j];
                    items[j] = temp;
                  }
                }
              }
            }
            """
        ast = self.compiler_state.parse(test_program)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())


    def test_recursive_factorial(self):
        test_program = """
            int factorial(int x);

            int main() {
              int x = 5;
              int result = factorial(x);

              return 0;
            }

            int factorial(int x) {
              if (x > 1) {
                return factorial(x - 1);
              } else {
                return 1;
              }
            }
            """
        ast = self.compiler_state.parse(test_program)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())


    def test_land_and_lor(self):
        test_program = """
            int main() {
              int x = 5;
              x = x && 5;

              return 0;
            }
            """
        ast = self.compiler_state.parse(test_program)
        source_tac, tac_str = ast.to_3ac()
        # print(source_tac)
        print('---------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())


