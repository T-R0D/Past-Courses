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

import unittest

from ticket_counting.ticket_counters import UUID_TICKETS, LABEL_TICKETS, INT_REGISTER_TICKETS, FLOAT_REGISTER_TICKETS
from compiler.compiler_state import CompilerState
from loggers.logger import Logger


class TestAst(unittest.TestCase):

    def setUp(self):
        self.compiler_state = CompilerState()
        self.enable_debug(False)
        UUID_TICKETS.next_value = 0
        LABEL_TICKETS.next_value = 0
        INT_REGISTER_TICKETS.next_value = 0
        FLOAT_REGISTER_TICKETS.next_value = 0

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

    def test_empty_file(self):
        data = ""
        ast = self.compiler_state.parse(data)
        print(ast)

    def test_plain_main(self):
        data = """
            int main()
            {
                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_simple_variable_declaration(self):
        data = """
            int main()
            {
                int i;
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_simple_variable_initialization(self):
        data = """
            int main()
            {
                int i = 5;
                int j = i;
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_const_declaration(self):
        data = """
            int main()
            {
                const int i = 5;
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_array_declaration(self):
        data = """
            int main()
            {
                int i[5];
            }
            """
        ast = self.compiler_state.parse(data)
        result = ast.to_graph_viz_str()
        print(result)

        import re

        expected_solution = \
            'digraph {\n' \
                '\t"FileAST\\\\n\d\d\d\d\d" -> {"FunctionDefinition\\\\nint main\\\\n\d\d\d\d\d"};\n' \
                '\t"FunctionDefinition\\\\nint main\\\\n\d\d\d\d\d" -> {"CompoundStatement\\\\n\d\d\d\d\d"};\n' \
                '\t"CompoundStatement\\\\n\d\d\d\d\d" -> {"ArrayDeclaration\\\\nint\[5\]\[5\] i\\\\n\d\d\d\d\d"};\n' \
                '\t"ArrayDeclaration\\\\nint\[5\]\[5\] i\\\\n\d\d\d\d\d" -> {};\n' \
            '}'

        m = re.match(expected_solution, result)
        print(m)
        self.assertTrue(True if m else False)

    def test_2d_array_declaration(self):
        data = """
            int main()
            {
                int i[5][7];
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_post_plus_plus(self):
        data = """
            int main()
            {
                int i = 0;
                int b = 0;
                b = i++;
                return 0;
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_lone_if(self):
        data = """
            int main()
            {
                int i;
                if (i == 5)
                {
                    i = 6;
                }
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

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
        print(ast.to_graph_viz_str())

    def test_if_elif_else(self):
        data = """
            int main()
            {
                int i;
                if (i == 5)
                {
                    i = 6;
                }
                else if(i == 6)
                {
                    i = 7;
                }
                else
                {
                    i = 5;
                }
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_simple_assign_const(self):
        data = """
            int main()
            {
                int g;
                char a;
                float p;

                g = 5;
                a = 2;
                p = 1.2;
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_simple_assign_var(self):
        data = """
            int main()
            {
                int g;
                int G;

                g = 5;
                G = g;
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_cast_in_binary_expression(self):
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


    def test_array_simple_assign(self):
        data = """
            int main()
            {
                int a[10];
                a[1] = 4;
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_array_simple_access(self):
        data = """
            int main()
            {
                int g;
                int a[10];
                a[1] = 4;
                g = a[1];
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_array_access_const_expr(self):
        data = """
            int main()
            {
                int g;
                int a[10];
                a[6] = 4;
                g = a[5 + 1];
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_array_access_var_expr(self):
        data = """
            int main()
            {
                int g;
                int a[10];
                a[1] = 4;
                g = a[1];
                g = a[g + 1];
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_array_twodim(self):
        data = """
            int main()
            {
                int b[10][10];
                b[1][1] = 5;
            }
            """

        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_for_loop_1(self):
        data = """
            int main()
            {
                int i;
                for(i = 0; ;) {}
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_for_loop_2(self):
        data = """
            int main()
            {
                int i;
                for(i = 0; ; i++) {}
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_for_loop_3(self):
        data = """
            int main()
            {
                int i;
                for(i = 0; i < 1; i++) {}
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_while_loop(self):
        data = """
            int main()
            {
                int i;
                while(i < 5){}
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_do_while_loop(self):
        data = """
            int main()
            {
                int i;
                do {} while (i > 10);
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_function_decl_top_impl_bottom(self):
        data = """
            int do_stuff();

            int do_stuff()
            {
                return 5;
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_function_decl_top_impl_bottom_call_middle(self):
        data = """
            int do_stuff();

            int main()
            {
                return do_stuff();
            }

            int do_stuff()
            {
                return 5;
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_function_parameters(self):
        # TODO: this one fails because we don't have the concept of a "dereference expression/operation", although
        # TODO: we aren't super worried about pointers for now.
        data = """
            int do_stuff(int* ret, int x)
            {
                *ret = x + x;
                return 5;
            }

            int main()
            {
                int* ptr;
                int num;
                return do_stuff(ptr, num);
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_function_def_on_top(self):
        data = """
            // Definition on top
            int do_stuff()
            {
                return 5;
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_function_def_on_top_call(self):
        data = """
            int do_stuff()
            {
                return 5;
            }

            int main()
            {
                return do_stuff();
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())

    def test_declare_const_and_var_types(self):
        data = """
            int main(){

                int x;
                int * y = 0;
                int z[10];
                const int i;
                float j;
                char k = 'a';
            }
            """
        ast = self.compiler_state.parse(data)
        print(ast.to_graph_viz_str())
