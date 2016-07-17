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

from compiler.compiler_state import CompilerState
from exceptions.compile_error import CompileError
from loggers.logger import Logger
from symbol_table.symbol import FunctionSymbol


class TestParser(unittest.TestCase):
    def setUp(self):
        self.debug = True
        self.compiler_state = CompilerState()

    def tearDown(self):
        self.compiler_state.teardown()
        self.compiler_state = None

    def test_plain_main(self):
        data = """
            int main()
            {
                return 0;
            }
            int i; // Needed to clone the symbol table at the correct time
            !!C
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'main', 0, 'int main()')

    def test_declare_primitive_variable(self):
        self.enable_parser_debugging()

        data = """
            int main() {
                int i;
                !!C
                return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'i', 2, 'int i')
        symbol, _ = symbol_table_clone.find('i')
        self.assertEqual(0, symbol.activation_frame_offset)

    def test_declare_and_assign_primitive_variable(self):
        data = """
            int main() {
                int i = 5;
                !!C
                return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'i', 2, 'int i')

    def test_declare_multiple_primitive_variable(self):
        # self.enable_parser_debugging()
        data = """
            int main() {
                int i, j, k;

                i = 0;
                !!C
                return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'i', 2, 'int i')
        self.check_correct_element(symbol_table_clone, 'j', 2, 'int j')
        self.check_correct_element(symbol_table_clone, 'k', 2, 'int k')

        i_symbol, _ = symbol_table_clone.find('i')
        j_symbol, _ = symbol_table_clone.find('j')
        k_symbol, _ = symbol_table_clone.find('k')

        self.assertEqual(0, i_symbol.activation_frame_offset)
        self.assertEqual(4, j_symbol.activation_frame_offset)
        self.assertEqual(8, k_symbol.activation_frame_offset)

    def test_modify_primitive_variable(self):
        self.enable_parser_debugging()

        data = """
            int main() {
                int i = 0;
                i += 5;
                !!C
                return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'i', 2, 'int i')

    def test_declare_pointer_variable(self):
        self.enable_parser_debugging()
        data = """
            int main() {
                int* i;
                i = 0;
                !!C
                return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'i', 2, 'int * i')

    def test_declare_deep_pointer_variable(self):
        data = """
            int main() {
                int*** i;
                !!C
                return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        print(symbol_table_clone)

        self.check_correct_element(symbol_table_clone, 'i', 2, 'int *** i')

    def test_declare_global_constant(self):
        self.enable_parser_debugging()
        data = """
            const int GLOBAL_CONSTANT = 5;

            int main() {
              int i = GLOBAL_CONSTANT;
              !!C
              return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'GLOBAL_CONSTANT', 0, 'const int GLOBAL_CONSTANT')
        self.check_correct_element(symbol_table_clone, 'i', 2, 'int i')

    def test_assign_to_immutable_variable_fails(self):
        with self.assertRaises(CompileError):
            data = """
                const int GLOBAL_CONSTANT = 5;

                int main() {
                  GLOBAL_CONSTANT = 0;
                  return 0;
                }
            """
            self.compiler_state.parse(data)

    def test_plain_if(self):
        data = """
            int main(int argc, char** argv)
            {
              if (1 != 1)
              {
                int wtf_result = -1;
                return wtf_result;
                !!C
              }

              return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]
        self.check_correct_element(symbol_table_clone, 'wtf_result', 3, 'int wtf_result')

    def test_ternary_operator(self):
        data = """
            int main(int argc, char** argv)
            {
               return 0 == 0 ? 1 : 0;
            }
           """
        self.compiler_state.parse(data)
        self.assertTrue(True, "No exceptions means a successful parse.")

    def test_plain_if_else(self):
        data = """
            int main(int argc, char** argv)
            {
                if (1 != 1)
                {
                    !!C
                    int wtf_result = -1;
                    return wtf_result;
                }
                else
                {
                    !!C
                    int i_guess_its_fine = 0;
                    return i_guess_its_fine;
                }
                return 0;
            }
        """
        print(data)
        self.compiler_state.parse(data)
        if_symbol_table_clone = self.compiler_state.cloned_tables[0]
        else_symbol_table_clone = self.compiler_state.cloned_tables[1]

        self.check_correct_element(if_symbol_table_clone, 'wtf_result', 3, 'int wtf_result')
        self.check_correct_element(else_symbol_table_clone, 'i_guess_its_fine', 3, 'int i_guess_its_fine')

    def test_lone_else_fails(self):

        with self.assertRaises(Exception):
            data = 'int main(int argc, char** argv) {\n' \
                   '  else {\n' \
                   '    int wtf = 0;\n' \
                   '    return wtf;\n' \
                   '  }\n' \
                   '\n' \
                   '  return 0;\n' \
                   '}\n' \
                   ''
            self.compiler_state.parse(data)

    def test_while_loop(self):
        data = """
            int main()
            {
              while (1) {}
              return 0;
            }
            int i; // Needed to clone the symbol table at the correct time
            !!C
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'main', 0, 'int main()')

    def test_for_loop(self):
        self.enable_parser_debugging()
        data = """
            int main()
            {
              int i;
              for (i = 0; i < 3; i++) {}
              !!C
              return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'i', 2, 'int i')

    def test_do_while_loop(self):
        data = """
            int main()
            {
              int i = 1;
              do {i++;} while(i);
              !!C
              return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'i', 2, 'int i')

    def test_cast_in_binary_expression(self):
        self.enable_parser_debugging()
        data = """
            int main()
            {
                int i = 5;
                float f = -4.5;

                i = (int) ((float) i + f);

                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_declare_array(self):
        self.enable_parser_debugging()
        data = """
            int main()
            {
              int my_array[10];
              !!C
              return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'my_array', 2, 'int my_array[10]')

    def test_declare_array_with_constant_expression_in_subscript(self):
        data = """
            int main()
            {
              int my_array[5 + 5];
              int i;
              !!C
              return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'my_array', 2, 'int my_array[10]')

    def test_access_array(self):
        self.enable_parser_debugging()
        data = """
            int main()
            {
              int i = 0;
              int my_array[10];

              int first_element = my_array[0];
              int some_other_element = my_array[i];

              !!C
              return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        print(symbol_table_clone)

        self.check_correct_element(symbol_table_clone, 'i', 2, 'int i')
        self.check_correct_element(symbol_table_clone, 'my_array', 2, 'int my_array[10]')
        self.check_correct_element(symbol_table_clone, 'first_element', 2, 'int first_element')
        self.check_correct_element(symbol_table_clone, 'some_other_element', 2, 'int some_other_element')

    def test_declare_function(self):
        self.enable_parser_debugging()
        data = """
            int do_stuff(char c);
            !!C

            int main()
            {
              return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        result, _ = symbol_table_clone.find('do_stuff')
        print(result, type(result))
        assert(isinstance(result, FunctionSymbol))
        p = result.named_parameters[0]
        print(p, type(p))
        print(p.type_specifiers)

        self.check_correct_element(symbol_table_clone, 'do_stuff', 0, 'int do_stuff(char c)')

    def test_declare_function_implementation(self):
        data = """
            int do_stuff(char c)
            {
                return c + c;
            }

            int main()
            {
              return 0;
            }
            !!C
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'do_stuff', 0, 'int do_stuff(char c)')

    def test_call_function(self):
        self.enable_parser_debugging()
        data = """
            int do_stuff(int c);
            !!C

            int main()
            {
              do_stuff(4);

              return 0;
            }

            int do_stuff(int c)
            {
                return c + c;
                !!C
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone_inner = self.compiler_state.cloned_tables[0]
        symbol_table_clone_outer = self.compiler_state.cloned_tables[1]

        self.check_correct_element(symbol_table_clone_inner, 'do_stuff', 0, 'int do_stuff(int c)')
        self.check_correct_element(symbol_table_clone_outer, 'c', 1, 'int c')

        symbol, _ = symbol_table_clone_outer.find('c')
        self.assertEqual(0, symbol.activation_frame_offset)

    def test_declare_string_literal_char_star(self):
        self.enable_parser_debugging()
        data = """
            char* literal_string = "hello there";
            !!C

            int main()
            {
              return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'literal_string', 0, 'char * literal_string')

    def test_declare_string_as_array(self):
        self.enable_parser_debugging()
        data = """
            int main()
            {
              char array_string[] = "hey";
              !!C
              return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'array_string', 2, 'char array_string[4]')

    def test_declare_segmented_string_literal(self):
        data = """
            char literal_string[] = "hello "
                                   "world";
            !!C
            int main() {
              return 0;
            }
        """
        self.compiler_state.parse(data)
        symbol_table_clone = self.compiler_state.cloned_tables[0]

        self.check_correct_element(symbol_table_clone, 'literal_string', 0, 'char literal_string[12]')

    def test_bubble_sort(self):
        # TODO: this test is failing because we are not handling pointers as though they were arrays and vice versa
        # TODO: perhaps we should change our test case? Maybe this is why Fred said pointers were hard...

        data = """
            void print(int list[], int size);
            void bubbleSort(int list[], int size);

            int main()
            {
               int list[10];
               int i;
               //srand(time(NULL));

               // create list
               for(i =0; i<10;i++)
               {
                   //list[i] = rand() % 10 + 1;
               }
               print(list, 10);

               // bubble sort
               bubbleSort(list, 10 );

               //printf( "Sorted " );
               print(list, 10);

               !!C

               // return
               return 0;

            }

            void bubbleSort(int list[], int size)
            {
               int i, j;
               int temp;
               int swapped;

               for( i = 0; i < size; i++)
               {

                  // swapped is false
                  swapped = 0;

                  for( j = 0; j < size - 1; j++)
                  {
                     if(list[j+1] < list[j])
                     {
                        temp = list[j];
                        list[j] = list[j+1];
                        list[j+1] = temp;
                        swapped = 1;
                     }
                  }

                  if (swapped == 0)
                  {
                     break;
                  }
               }
            }

            void print(int list[], int size)
            {
               int i;
               //printf("List is: ");

               for(i =0; i < size; i++)
               {
                  //printf( "%d ", list[i] );
               }
               //printf("");
            }
        """
        self.compiler_state.parse(data)
        #symbol_table_clone = self.compiler_state.cloned_tables[0]

        # TODO: Determine what scopes we want to test here
        #self.check_correct_element(symbol_table_clone, '', 1, '')
        self.assertTrue(True, 'No exceptions = Parser successfully parsed.')

    def test_recursive_factorial(self):
        data = """
            long int recur_Fact(int number);

            int main() {
              int number;
              long int fact;

              //printf( "Enter number to get factorial of: ");
              //scanf( "%d", &number );

              fact = recur_Fact(number);

              //printf("Factorial of %d is:  %ld", number, fact);

              return 0;
            }

            long int recur_Fact( int number) {
              // base case
              if(number <= 0)
                return 1;

              // recursive case
              else if( number > 1 ) {
                return number*recur_Fact(number-1);
              }
            }
        """
        self.compiler_state.parse(data)
        #symbol_table_clone = self.compiler_state.cloned_tables[0]

        # TODO: Determine what scopes we want to test here
        #self.check_correct_element(symbol_table_clone, '', 1, '')

        self.assertTrue(True, 'No exceptions = Parser successfully parsed.')

    def test_iterative_factorial(self):
        data = """
            long int iter_Fact(int number);

            int main()
            {
                int number;
                long int fact;

                // printf("Enter number to get factorial of: ");
                // scanf( "%d", &number );

                fact = iter_Fact(number);

                // printf("Factorial of %d is:  %ld ", number, fact);

                return 0;
            }

            long int iter_Fact(int number)
            {
                int i;
                long int fact = 1;

                if( i < 0)
                {
                    return 1;
                }

                for( i = number; i > 0; i --)
                {
                    fact = fact*i;
                }
                return fact;
            }
        """
        self.compiler_state.parse(data)
        #symbol_table_clone = self.compiler_state.cloned_tables[0]

        # TODO: Determine what scopes we want to test here
        #self.check_correct_element(symbol_table_clone, '', 1, '')

        self.assertTrue(True, 'No exceptions = Parser successfully parsed.')

    # TODO: don't put a lot of emphasis on bad cases until things are strong with the good cases
    def test_malformed_main_fails(self):
        with self.assertRaises(Exception):
            data = 'badmain(] {return 0;}'
            self.compiler_state.parse(data)


    # def test_declare_typedef(self):
    #     self.enable_parser_debugging()
    #     data = """
    #     typedef int GlorifiedInt;
    #     !!C
    #
    #     int main() {
    #       return 0;
    #     }
    #     """
    #     self.compiler_state.parse(data)
    #     symbol_table_clone = self.compiler_state.cloned_tables[0]
    #
    #     # TODO: How are we handling typedefs?
    #     self.check_correct_element(symbol_table_clone, 'GlorifiedInt', 0, 'typedef int GlorifiedInt')

    # def test_declare_typedef_and_use_typedef_in_variable_declaration(self):
    #     self.enable_parser_debugging()
    #
    #     data = """
    #     typedef int GlorifiedInt;
    #
    #     int main() {
    #       GlorifiedInt i = 3;
    #       return 0;
    #       !!C
    #     }
    #     """
    #     self.compiler_state.parse(data)
    #     symbol_table_clone = self.compiler_state.cloned_tables[0]
    #
    #     # TODO: How are we handling typedefs?
    #     self.check_correct_element(symbol_table_clone, 'GlorifiedInt', 0, 'typedef int GlorifiedInt')
    #     self.check_correct_element(symbol_table_clone, 'i', 1, 'GlorifiedInt')

    # def test_declare_struct(self):
    #     data = """
    #         !!C
    #         struct Pixel {
    #             char r;
    #             char g;
    #             char b;
    #             !!C
    #         };
    #
    #         int main() {
    #           struct Pixel pixel;
    #           pixel.r = 255;
    #           pixel.g = 255;
    #           pixel.b = 255;
    #           !!C
    #           return 0;
    #         }
    #     """
    #
    #     self.compiler_state.parse(data)
    #     symbol_table_clone = self.compiler_state.cloned_tables[0]
    #
    #     # TODO: How are we handling structs?
    #     self.check_correct_element(symbol_table_clone, 'Pixel', 0, 'struct Pixel')
    #     self.check_correct_element(symbol_table_clone, 'r', 1, 'char r')
    #     self.check_correct_element(symbol_table_clone, 'g', 1, 'char g')
    #     self.check_correct_element(symbol_table_clone, 'b', 1, 'char b')
    #     self.check_correct_element(symbol_table_clone, 'pixel', 1, 'struct Pixel pixel')
    #
    #
    # def test_declare_function_pointer_typedef(self):
    #     data = """
    #     typedef int (*add_callback)(int a, int b);
    #
    #     int add_two(int a, int b, add_callback callback);
    #
    #     int normal_add(int a, int b);
    #     int weird_add(int a, int b);
    #
    #     int main() {
    #       int x;
    #       int y;
    #
    #       x = add_two(1, 2, normal_add);
    #       y = add_two(1, 2, weird_add);
    #
    #       !!C
    #       return 0;
    #     }
    #
    #     int add_two(int a, int b, add_callback callback) {
    #         return callback(a, b);
    #     }
    #
    #     int normal_add(int a, int b) {
    #         return a + b;
    #     }
    #
    #     int weird_add(int a, int b) {
    #         return (a + b) % 4;
    #     }
    #     """
    #     self.compiler_state.parse(data)
    #     symbol_table_clone = self.compiler_state.cloned_tables[0]
    #
    #     # TODO: Function Pointers??
    #     self.check_correct_element(symbol_table_clone, 'x', 1, 'int x')

    def test_super_function_testing(self):
        data = """
            !!C void do_stuff(int* array);

            int main()
            {
                return 0;
            }

            void do_stuff(int* array)
            {
                int* i;
                do_stuff(i);
            }
        """

        self.compiler_state.parse(data)

        symbol_table = self.compiler_state.cloned_tables[0]
        print(symbol_table)
        x, y = symbol_table.find('do_stuff')
        self.check_correct_element(symbol_table, 'do_stuff', 0, 'void do_stuff(int * array)')

    def test_super_memory_allocation(self):
        data = """
            char g_char;
            int g_int;
            float g_float;

            void do_stuff(char a, int b) {
                int c;
                float d;

                !!C
                return a + b;
            }

            int main()
            {
                int i = do_stuff('a', 2);
                return 0;
            }
            !!C
        """

        self.compiler_state.parse(data)

        function_symbol_table, global_symbol_table = self.compiler_state.cloned_tables[0:2]

        # g_char, _ = global_symbol_table.find('g_char')
        # self.assertEqual(0x10010000, g_char.global_memory_location)
        # g_int, _ = global_symbol_table.find('g_int')
        # self.assertEqual(0x10010004, g_int.global_memory_location)
        # g_float, _ = global_symbol_table.find('g_float')
        # self.assertEqual(0x10010008, g_float.global_memory_location)

        f_a, _ = function_symbol_table.find('a')
        self.assertEqual(0, f_a.activation_frame_offset)
        f_b, _ = function_symbol_table.find('b')
        self.assertEqual(4, f_b.activation_frame_offset)
        f_c, _ = function_symbol_table.find('c')
        self.assertEqual(8, f_c.activation_frame_offset)
        f_d, _ = function_symbol_table.find('d')
        self.assertEqual(12, f_d.activation_frame_offset)

    def enable_parser_debugging(self):
        if self.debug:
            self.compiler_state.get_parser_logger().add_switch(Logger.PRODUCTION)
            self.compiler_state.get_parser_logger().add_switch(Logger.INFO)
            self.compiler_state.get_parser_logger().add_switch(Logger.SOURCE)

    def check_correct_element(self, symbol_table_clone, check_value, check_scope, check_string):
        found_symbol, in_scope = symbol_table_clone.find(check_value)
        self.assertEqual(check_scope, in_scope, "The symbol was not found in the expected scope")
        self.assertEqual(check_string, str(found_symbol), "The symbols don't match")
