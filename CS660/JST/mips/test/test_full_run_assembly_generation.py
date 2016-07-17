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


#~~~
# The purpose of this file is to have the plainest possible test cases for presentation purposes. This file should
# not have experimental or correctness verification tests, but rather succinct, clean test cases or well-thought out
# multi-feature tests that are directly intended for demonstration of the compiler's capabilities.
#
# Unlike other files, this file should not produce anything other than what will end up in a MIPS file.
#~~~


import unittest
import compiler.compiler_state as compiler
import mips.generation as generation

import loggers.logger as log


class TestFullRunAssemblyGeneration(unittest.TestCase):
    def setUp(self):
        self.compiler_state = compiler.CompilerState(print_productions=False)
        self.enable_debug(False)

        self.generator = generation.MipsGenerator(self.compiler_state, inject_source = False, inject_3ac=False)

    def tearDown(self):
        self.compiler_state.teardown()
        self.compiler_state = None

    def enable_debug(self, enable, productions=True, source=False):
        if enable:
            prod_logger = self.compiler_state.get_parser_logger()

            prod_logger.add_switch(log.Logger.INFO)
            if productions:
                prod_logger.add_switch(log.Logger.PRODUCTION)

            if source:
                prod_logger.add_switch(log.Logger.SOURCE)

    def test_plain_main(self):
        data = """
            int main() {
                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/plain_main.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/plain_main.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()

    def test_local_variable_declaration_and_assignment(self):
        data = """
            int main() {
                int local_variable;

                // print the garbage that will be in the variable
                print_int(local_variable);      // will most likely see 0 but could be different since its garbage

                // perform an assignment and print to show that the
                // value was assigned
                local_variable = 123;           // expect to see 123
                print_int(local_variable); print_char('\\n');

                // assign another value to show that it can be overwritten
                local_variable = 126;           // expect to see 126
                print_int(local_variable);

                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/local_variable_declaration_and_assignment.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/local_variable_declaration_and_assignment.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()

    def test_local_variable_addition(self):
        data = """
            int main() {
                int local_variable = 1;
                int other_variable = 9;
                int third_variable = 5;

                // print the values that will be in the variables
                print_int(local_variable);  // expect to see 1
                print_int(other_variable);  // expect to see 9
                print_int(third_variable);  // expect to see 5

                // perform the addition
                print_int(local_variable + other_variable);         // expect to see 10
                third_variable = local_variable + other_variable;
                print_int(third_variable);                          // expect to see 10, again

            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        # i = 0;
        # for item in source_tac:
        #     if i% 3 == 0:
        #         print('\n')
        #     print(item)
        #
        # print('\n\n\n\n')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/local_variable_addition.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/local_variable_addition.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()

    def test_global_variables_declaration_and_assignment(self):
        data = """
            const int GLOBAL_CONST = 4;
            int GLOBAL_VAR = 2;

            int main() {

                // print the values that will be in the variables
                print_int(GLOBAL_CONST);    // expect to see 4
                print_int(GLOBAL_VAR);      // expect to see 2

                // perform the assignment
                GLOBAL_VAR = GLOBAL_CONST;
                print_int(GLOBAL_VAR);      // expect to see 4

                return 0;
            }


            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        # #TODO: Take out debug after fixing test case issues
        # i = 0;
        # for item in source_tac:
        #     if i% 3 == 0:
        #         print('\n')
        #     print(item)
        #
        # print('\n\n\n\n')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/global_variables_declaration_and_assignment.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/global_variables_declaration_and_assignment.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()

    def test_if_elif_else(self):
        data = """
            int main() {
                int i = 0;

                // FizzBuzz
                for( i = 1; i <= 30; i++) {
                   //FizzBuzz
                   if( i % 3 == 0){

                        //FizzBuzz
                        if( i % 5 == 0 ){
                            // expect to see this at 15 and 30
                            print_int(i); print_char(':'); print_char(' ');
                            print_char('f'); print_char('b');
                            print_char('\\n');
                        }

                        //Fizz
                        else {
                           // expect to see this at 3,6,9,12,18,21,24,27
                           print_int(i); print_char(':'); print_char(' ');
                           print_char('f');
                           print_char('\\n');
                        }

                   }
                   // Buzz
                   else if( i % 5 == 0) {
                       // expect to see this at 5,10,15,20,25
                       print_int(i); print_char(':'); print_char(' ');
                       print_char('b');
                       print_char('\\n');
                   }
                   // Number
                   else {
                       // expect to see all other numbers except those mentioned above
                       print_int(i); print_char('\\n');
                   }
                }

                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        print(i)
        print('----------------------------')

        self.generator.inject_source = True
        self.generator.inject_3ac = True
        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/if_elif_else.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/if_elif_else.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()

    def test_all_three_loop_types(self):
        data = """
            int main() {
                /**
                 * The following test is an interesting composition of loops.
                 * First the _do-while_ runs through all of its iterations.
                 * Then the _while_ loop runs, but with each of its iterations, a "do" is
                 * forced, so we see a _do-while_ iteration along with each _while_ iteration.
                 * Finally, the _for_ is allowed to progress without additional _while_ iterations.
                 * This is further discussed in the comments on each line
                 */

                int i = 0;
                int j = 10;
                int k = 15;

                // test for loop
                for( i = 0; i <= 5; i ++ ) {

                    // test while loops
                    while( j <= 15 ) {

                        // test do while loops
                        do{
                            // expect to see 15-20, then 21-25 interspersed with numbers from js
                            print_char('k'); print_char(':'); print_char(' '); print_int(k);
                            print_char('\\n');
                            k++;
                        } while( k <= 20 );

                        // expect to see 10-15 interspersed with the numbers from the ks do
                        print_char('j'); print_char(':'); print_char(' '); print_int(j);
                        print_char('\\n');

                        j++;
                    }

                    // expect to see 0-5
                    print_char('i'); print_char(':'); print_char(' '); print_int(i);
                    print_char('\\n');
                }

                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        print(i)
        print('----------------------------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/all_three_loops.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/all_three_loops.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()

    def test_while_loops_nested(self):
        data = """
            int main() {

                int i = 0;
                int j = 10;
                int k = 15;

                // test while loops
                while( i <= 5 ) {
                    while( j <= 15 ) {
                        while( k <= 20 ) {
                            print_int(k);   // expect to see 15-20
                            k++;
                        }
                        print_int(j);   // expect to see 10-15
                        j++;
                    }
                    print_int(i);       // expect to see 0-5
                    i++;
                }

                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/while_loops_nested.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/while_loops_nested.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()

    def test_do_while_loops_nested(self):
        data = """
            int main() {

                int l = 20;
                int m = 25;

                // test do while loops
                do{
                  print_int(l);     // expect to see 20, then the m's, then 21-24 interspersed with the m's do
                  l++;

                  do{
                    print_int(m);   // expect to see 25-29 then 30-33 interspersed with numbers from l's do
                    m++;

                  }while ( m < 30 );

                }while (l < 25);

                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())


        fout = open("../../res/c_files/do_while_loops_nested.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/do_while_loops_nested.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()

    def test_for_loops_nested(self):
        data = """
            int main() {

                int n = 0;
                int p = 0;

                // test for loops
                for( n = 0; n < 5; n++) {
                    for( p = 0; p < 5; p ++ ) {
                        print_int(p);   // expect to see 0-4 then 0-4 after each increment of p
                    }
                    print_int(n); // expect to see 0-4 with the 0-4 from the p's after 0,1,2,3
                }
                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/for_loops_nested.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/for_loops_nested.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()

    def test_array_declarations_and_manipulation(self):
        data = """
            int main() {

                int i[3];
                int j[2][2];
                int k[2][2][2][2][2];
                int s;

                // 1-D manipulation
                i[0] = 2;
                i[2] = i[0];
                for( s = 0; s < 3; s++) {
                    print_int(i[s]);    // expect to see 2, 0, 2
                }

                // 2-D manipulation
                j[0][0] = 20;
                print_int(j[0][0]); // expect to see 20
                j[1][1] = j[0][0];
                print_int(j[1][1]); // expect to see 20


                // 5-D manipulation
                k[1][0][1][0][1] = 45;
                print_int(k[1][0][1][0][1]);    // expect to see 45
                k[0][0][0][0][1] = k[1][0][1][0][1];
                print_int(k[0][0][0][0][1]);    // expect to see 45


                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/array_declaration_and_manipulation.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/array_declaration_and_manipulation.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()

    def test_function_call(self):
        data = """
            int foo( int a, char b);

            int main()
            {
                int i = foo(1, 'a');
                print_int(i);        // expect to see 123
                return 0;
            }

            int foo( int a, char b)
            {
                print_int(a);        // expect to see 1
                return 123;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        #TODO: Take out debug after fixing test case issues
        # i = 0;
        # for item in source_tac:
        #     if i% 3 == 0:
        #         print('\n')
        #     print(item)
        #
        # print('\n\n\n\n')


        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/function_call.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/function_call.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()


    def test_post_and_pre_increment(self):
        data = """
            int main() {

                int i = 1;
                int j = 1;

                j = i++;
                print_int(j);   // expect to see 1
                print_int(i);   // expect to see 2

                j = ++i;
                print_int(j);   // expect to see 3
                print_int(i);   // expect to see 3


                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/post_and_pre_increment.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/post_and_pre_increment.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()

    def test_arrays_in_functions(self):
        data = """
            int foo(int a[][])
            {
                int i, j;
                for(i = 0; i < 3; ++i)
                {
                    for(j = 0; j < 7; ++j)
                    {
                        print_char(i + '0');
                        print_char(' ');
                        print_char(j + '0');
                        print_char(' ');
                        print_char(' ');
                        print_char(' ');
                        print_int(a[i][j]);
                        print_char('\\n');
                    }
                }
            }

            int main() {

                int b[3][7];
                int i, j;
                for(i = 0; i < 3; ++i)
                {
                    for(j = 0; j < 7; ++j)
                    {
                        b[i][j] = (i*7) + j;
                    }
                }

                foo(b);

                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        self.generator.inject_source = True

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/arrays_in_functions.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/arrays_in_functions.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()

    def test_constant_folding(self):
        data = """

            const int C = 4 + 4;

            int main() {
                // expect to find a li with 14 in the assembly instructions
                // since we have constant folding working correctly
                int n = C - 2 + 4 * 2;

                // expect to see the 14 printed to show its loaded into n correctly
                print_int(n);

                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        print(i)
        print('--------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/constant_folding.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/constant_folding.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()


    def test_binary_operators(self):
        data = """
            int main() {

                int i = 0;
                int j = 0;
                int k = 0;

                i = i + 10; print_int(i); print_char('\\n'); // prints 10
                i = i - 2;  print_int(i); print_char('\\n'); // prints 8
                i = i * 2;  print_int(i); print_char('\\n'); // prints 16
                i = i / 4;  print_int(i); print_char('\\n'); // prints 4
                i = i % 3;  print_int(i); print_char('\\n'); // prints 1
                print_char('\\n');

                j = i++;
                // prints 1 and 2
                print_int(j); print_char(' '); print_int(i); print_char('\\n');

                j = ++i;
                // prints 3 and 3
                print_int(j); print_char(' '); print_int(i); print_char('\\n');
                print_char('\\n');

                j = i--;
                // prints 3 and 2
                print_int(j); print_char(' '); print_int(i); print_char('\\n');

                j = --i;
                // prints 1 and 1
                print_int(j); print_char(' '); print_int(i); print_char('\\n');
                print_char('\\n');

                j += i;
                // prints 2
                print_int(j); print_char('\\n');

                j -= i;
                // prints 1
                print_int(j); print_char('\\n');
                print_char('\\n');

                k = i = j;
                // prints 1 1 1
                print_int(k); print_char(' '); print_int(i); print_char(' '); print_int(j);
                print_char('\\n');
                print_char('\\n');

                i = i && 0;
                print_int(i); print_char('\\n'); // prints 0

                i = 1 && 1;
                print_int(i); print_char('\\n'); // prints 1
                print_char('\\n');

                j = i || 5;
                print_int(j); print_char('\\n'); // prints 1

                j = 0 || 0;
                print_int(j); print_char('\\n'); // prints 0

                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        print(i)
        print('--------------------------------')

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/binary_operators.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/binary_operators.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()




#####################
# Test cases that are not put to .c or .asm files
#####################

    def test_parse_error(self):
        data = """
            int main() {

                int n = 0;

                // test for loops
                for( n = 0; n < 5; n++) {
                   print_int(n); */
                }
                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())




# # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # #
#     TESTING THE BIG THREEEEE
# # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # #


    def test_bubble_sort(self):
        data = """
            const int N_ITEMS = 5;

            int main() {
              int i, j, temp;
            // int things[N_ITEMS] = {5, 1, 4, 3, 2};

              int things[N_ITEMS];
              things[0] = 5;
              things[1] = 1;
              things[2] = 4;
              things[3] = 3;
              things[4] = 2;

              print_int(things[0]);  // expect to see 5
              print_char('\\n');
              print_int(things[1]);  // expect to see 1
              print_char('\\n');
              print_int(things[2]);  // expect to see 4
              print_char('\\n');
              print_int(things[3]);  // expect to see 3
              print_char('\\n');
              print_int(things[4]);  // expect to see 2
              print_char('\\n');

              for (i = 0; i < N_ITEMS; i++) {
                for (j = i; j < N_ITEMS; j++) {
                  if (things[i] < things[j]) {
                    temp = things[i];
                    things[i] = things[j];
                    things[j] = temp;
                  }
                }
              }

              print_char('\\n');
              print_int(things[0]);  // expect to see 5
              print_char('\\n');
              print_int(things[1]);  // expect to see 4
              print_char('\\n');
              print_int(things[2]);  // expect to see 3
              print_char('\\n');
              print_int(things[3]);  // expect to see 2
              print_char('\\n');
              print_int(things[4]);  // expect to see 1

              return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()
        print(i);

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/bubble_sort.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/bubble_sort.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()

    def test_matrix_multiplication(self):
        data = """
           const int ARRAY_DIM = 2;

            // hard code dimensions for simplicity
            int matrix_multiply(int C[ARRAY_DIM][ARRAY_DIM], int A[ARRAY_DIM][ARRAY_DIM], int B[ARRAY_DIM][ARRAY_DIM]);
            int print_matrix(int C[ARRAY_DIM][ARRAY_DIM]);

            int main() {
              int i, j, k;
              int sum;
              int A[ARRAY_DIM][ARRAY_DIM], B[ARRAY_DIM][ARRAY_DIM], C[ARRAY_DIM][ARRAY_DIM];

              for (i = 0; i < ARRAY_DIM; i++) {
                for (j = 0; j < ARRAY_DIM; j++) {
                  A[i][j] = B[i][j] = 2;
                }
              }

              // matrix_multiply
              matrix_multiply(C, A, B);

              // print_matrix
              print_matrix(C);

              return 0;
            }

            int matrix_multiply(int C[ARRAY_DIM][ARRAY_DIM], int A[ARRAY_DIM][ARRAY_DIM], int B[ARRAY_DIM][ARRAY_DIM])
            {
              int i, j, k, sum;

              // matrix_multiply
              for (i = 0; i < ARRAY_DIM; i++) {
                 for (j = 0; j < ARRAY_DIM; j++) {
                    sum = 0;
                    for( k=0; k < ARRAY_DIM; k++) {
                        sum = sum + A[i][k] * B[k][j];
                    }
                    C[i][j] = sum;
                 }
              }
            }

            int print_matrix( int C[ARRAY_DIM][ARRAY_DIM])
            {
              int i, j;

              // print_matrix
              for (i = 0; i < ARRAY_DIM; i++) {
                    for (j = 0; j < ARRAY_DIM; j++) {
                        print_int(C[i][j]);     // expect to see four 8's
                        print_char(' ');
                    }
                    print_char('\\n');
              }
            }

            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/matrix_multiplication.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/matrix_multiplication.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()


    def test_recursive_factorial(self):
        data = """
            int factorial(int x);

            int main() {
              int x = 5;
              int result = factorial(x);
              print_int(x);       // expect to see 5
              print_char('\\n');
              print_int(result);  // expect to see 120

              return 0;
            }

            int factorial(int x) {
              if (x > 1) {
                return x*factorial(x - 1);
              } else {
                return 1;
              }
            }
            """
        ast = self.compiler_state.parse(data)
        source_tac, i = ast.to_3ac()
        print(i)

        self.generator.load(source_tac)
        self.generator.translate_tac_to_mips()
        print(self.generator.dumps())

        fout = open("../../res/c_files/recursive_factorial.c", 'w')
        fout.write(data)
        fout.close()

        fout = open("../../res/asm_files/recursive_factorial.asm", 'w')
        fout.write(self.generator.dumps())
        fout.close()


