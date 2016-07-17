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
import itertools
from ply import lex
from compiler.compiler_state import CompilerState
from scanning.jst_lexer import JSTLexer


class TestLexer(unittest.TestCase):

    SIMPLE_MAIN_START_TOKEN_TYPES = ['INT', 'ID', 'LPAREN', 'RPAREN', 'LBRACE']
    SIMPLE_MAIN_END_TOKEN_TYPES = ['RETURN', 'ICONST', 'SEMI', 'RBRACE']
    SIMPLE_MAIN_TOKEN_TYPES = SIMPLE_MAIN_START_TOKEN_TYPES + SIMPLE_MAIN_END_TOKEN_TYPES

    DECLARE_INT_TOKEN_TYPES = ['INT', 'ID', 'EQUALS', 'ICONST', 'SEMI']
    DECLARE_CHAR_TOKEN_TYPES = ['CHAR', 'ID', 'EQUALS', 'CCONST', 'SEMI']
    DECLARE_FLOAT_TOKEN_TYPES = ['FLOAT', 'ID', 'EQUALS', 'FCONST', 'SEMI']

    TEST_VAR_TOKEN_TYPES = SIMPLE_MAIN_START_TOKEN_TYPES + DECLARE_INT_TOKEN_TYPES + SIMPLE_MAIN_END_TOKEN_TYPES
    TEST_ILLEGAL_TOKEN_TYPES = SIMPLE_MAIN_START_TOKEN_TYPES + DECLARE_INT_TOKEN_TYPES + ['CHAR', 'EQUALS', 'CCONST', 'SEMI'] + SIMPLE_MAIN_END_TOKEN_TYPES
    TEST_GLOBAL_TOKEN_TYPES = ['CONST', 'INT', 'ID', 'EQUALS', 'ICONST', 'SEMI'] + SIMPLE_MAIN_TOKEN_TYPES
    TEST_ARRAY_TOKEN_TYPES = SIMPLE_MAIN_START_TOKEN_TYPES + ['INT', 'ID', 'LBRACKET', 'ICONST',  'RBRACKET', 'SEMI'] + SIMPLE_MAIN_END_TOKEN_TYPES
    TEST_CONST_TOKEN_TYPES = SIMPLE_MAIN_START_TOKEN_TYPES + DECLARE_INT_TOKEN_TYPES + DECLARE_FLOAT_TOKEN_TYPES + DECLARE_CHAR_TOKEN_TYPES + ['ID','LPAREN','SCONST','RPAREN','SEMI'] + SIMPLE_MAIN_END_TOKEN_TYPES
    TEST_FUNCTION_TOKEN_TYPES = ['INT', 'ID', 'LPAREN', 'CHAR', 'ID', 'RPAREN', 'SEMI'] + SIMPLE_MAIN_START_TOKEN_TYPES + ['ID', 'LPAREN', 'CCONST', 'RPAREN', 'SEMI'] + SIMPLE_MAIN_END_TOKEN_TYPES + ['INT', 'ID', 'LPAREN', 'CHAR', 'ID', 'RPAREN', 'LBRACE', 'RETURN', 'ID', 'PLUS', 'ID', 'SEMI', 'RBRACE']
    TEST_BANGBANGS_TOKEN_TYPES = ['INT', 'ID', 'LPAREN', 'CHAR', 'ID', 'RPAREN', 'SEMI'] + SIMPLE_MAIN_START_TOKEN_TYPES + ['ID', 'LPAREN', 'CCONST', 'RPAREN', 'SEMI'] + SIMPLE_MAIN_END_TOKEN_TYPES + ['INT', 'ID', 'LPAREN', 'CHAR', 'ID', 'RPAREN', 'LBRACE', 'RETURN', 'ID', 'PLUS', 'ID', 'SEMI', 'RBRACE']
    TEST_STRUCT_TOKEN_TYPES = ['STRUCT', 'ID', 'LBRACE', 'CHAR', 'ID', 'SEMI', 'CHAR', 'ID', 'SEMI', 'CHAR', 'ID', 'SEMI', 'RBRACE', 'SEMI'] + SIMPLE_MAIN_TOKEN_TYPES

    #NOTE: Don't have any RE for Ptrs!!!! - only tokenize them as MULT ID ....
    #NOTE: Also, TYPEID is never used here....
    TEST_FPTR_TOKEN_TYPES = ['TYPEDEF','INT','LPAREN','TIMES','ID','RPAREN','LPAREN','INT','ID','COMMA','INT','ID','RPAREN','SEMI'] + ['INT','ID','LPAREN','INT','ID','COMMA','INT','ID','COMMA','ID','ID','RPAREN','SEMI'] + ['INT','ID','LPAREN','INT','ID','COMMA','INT','ID','RPAREN','SEMI'] + ['INT','ID','LPAREN','INT','ID','COMMA','INT','ID','RPAREN','SEMI'] + SIMPLE_MAIN_START_TOKEN_TYPES + ['INT','ID','SEMI','INT','ID','SEMI'] + ['ID','EQUALS','ID','LPAREN','ICONST','COMMA','ICONST','COMMA','ID','RPAREN','SEMI'] + ['ID','EQUALS','ID','LPAREN','ICONST','COMMA','ICONST','COMMA','ID','RPAREN','SEMI'] + SIMPLE_MAIN_END_TOKEN_TYPES + ['INT','ID','LPAREN','INT','ID','COMMA','INT','ID','COMMA','ID','ID','RPAREN','LBRACE','RETURN','ID','LPAREN','ID','COMMA','ID','RPAREN','SEMI','RBRACE'] + ['INT','ID','LPAREN','INT','ID','COMMA','INT', 'ID', 'RPAREN','LBRACE','RETURN','ID','PLUS','ID','SEMI','RBRACE'] + ['INT','ID','LPAREN','INT','ID','COMMA','INT', 'ID', 'RPAREN','LBRACE','RETURN','LPAREN','ID','PLUS','ID','RPAREN','MOD','ICONST','SEMI','RBRACE']


    TEST_SYMBOLS_TOKEN_TYPES = ['PLUS','MINUS','TIMES','DIVIDE','MOD','OR','AND','NOT','XOR','LSHIFT','RSHIFT','LOR','LAND','LNOT','LT','LE','GT','GE','EQ','NE','EQUALS','TIMESEQUAL','DIVEQUAL','MODEQUAL','PLUSEQUAL','MINUSEQUAL','LSHIFTEQUAL','RSHIFTEQUAL','ANDEQUAL','XOREQUAL','OREQUAL','PLUSPLUS','MINUSMINUS','ARROW','CONDOP','LPAREN','RPAREN','LBRACKET','RBRACKET','LBRACE','RBRACE','COMMA','PERIOD','SEMI','COLON','ELLIPSIS']
    TEST_RESERVED_TOKEN_TYPES = ['AUTO','BREAK','CASE','CHAR','CONST','CONTINUE','DEFAULT','DO','DOUBLE','ELSE','ENUM','EXTERN','FLOAT','FOR','GOTO','IF','INT','LONG','REGISTER','RETURN','SHORT','SIGNED','SIZEOF','STATIC','STRUCT','SWITCH','TYPEDEF','UNION','UNSIGNED','VOID','VOLATILE','WHILE']

    def setUp(self):
        self.compiler_state = CompilerState()
        lexer = JSTLexer(self.compiler_state)
        self.scanner = lex.lex(module=lexer)

    def tearDown(self):
        self.lexer = None

    def test_plain_main(self):
        data = """int main() {return 0;}"""
        self.compare_token_output(data, expected_token_types=TestLexer.SIMPLE_MAIN_TOKEN_TYPES)

    def test_regular_comments_get_eaten(self):
        data = """
            // this is a comment
            int main() {return 0;}
        """
        self.compare_token_output(data, expected_token_types=TestLexer.SIMPLE_MAIN_TOKEN_TYPES)

    def test_regular_comments_after_code_get_eaten(self):
        data = """
            int main() {// this is a comment
              return 0;
            }
        """
        self.compare_token_output(data, expected_token_types=TestLexer.SIMPLE_MAIN_TOKEN_TYPES)

    def test_block_comments_get_eaten(self):
        data = """
            /* This is a block comment */
            int main() {return 0;}
        """
        self.compare_token_output(data, expected_token_types=TestLexer.SIMPLE_MAIN_TOKEN_TYPES)

    def test_block_comments_within_code_get_eaten(self):
        data = """int main(/* This is a block comment */) {return 0;}"""
        self.compare_token_output(data, expected_token_types=TestLexer.SIMPLE_MAIN_TOKEN_TYPES)

    def test_mulitiline_block_comments_get_eaten(self):
        data = """
        int main() {
          /**
           * Input: None
           * Returns: Error status of program execution.
           */
          return 0;
        }"""
        self.compare_token_output(data, expected_token_types=TestLexer.SIMPLE_MAIN_TOKEN_TYPES)

    def test_declare_var(self):
        data = """
            int main() {
                int i = 0;
                return 0;
            }
            """
        self.compare_token_output(data, expected_token_types=TestLexer.TEST_VAR_TOKEN_TYPES)

    def test_illegal_character(self):
        with self.assertRaisesRegex(Exception, "Illegal token: 사 = 'E';"):
            data = """
            int main() {
                int i = 0;
                char 사 = 'E';
                return 0;
            }
            """
            self.scanner.input(data)
            while True:
                tok = self.scanner.token()
                print(tok)
                if not tok:
                    break

    def test_declare_global_constant(self):
        data = """
        const int GLOBAL_CONSTANT = 5;

        int main() {
          return 0;
        }
        """
        self.compare_token_output(data, expected_token_types=TestLexer.TEST_GLOBAL_TOKEN_TYPES)

    def test_declare_array(self):
        data = """
        int main() {
          int my_array[10];
          return 0;
        }
        """
        self.compare_token_output(data, expected_token_types=TestLexer.TEST_ARRAY_TOKEN_TYPES)

    def test_block_comments(self):
        data = """
        /* this is a comment */
        int main() {

          int i = 0;

          /* this is another comment */
          return 0;
        }
        """
        self.compare_token_output(data, expected_token_types=TestLexer.TEST_VAR_TOKEN_TYPES)

    def test_const_tokens(self):
        data = """
            int main() {
                int i = 7;
                float j = 1.123;
                char f = 'k';
                printf("STRINGCONSTTT");
                return 0;
            }
            """
        self.compare_token_output(data, expected_token_types=TestLexer.TEST_CONST_TOKEN_TYPES)

    def test_declare_and_call_function(self):
        data = """
            int do_stuff(char c);

            int main() {
              do_stuff('f');
              return 0;
            }

            int do_stuff(char c) {
                return c + c;
            }
        """
        self.compare_token_output(data, expected_token_types=TestLexer.TEST_FUNCTION_TOKEN_TYPES)

    def test_bang_bang_flags(self):
        data = """
            int do_stuff(char c);

            int main() {
              do_stuff('f');

              return 0;
            }

            int do_stuff(char c) {
                !!S
                !!C
                !!C
                return c + c;
             }
        """
        self.compare_token_output(data, expected_token_types=TestLexer.TEST_BANGBANGS_TOKEN_TYPES)

    def test_declare_struct(self):
        data = """
            struct Pixel {
                char r;
                char g;
                char b;
            };

            int main() {
              return 0;
            }
        """
        self.compare_token_output(data, expected_token_types=TestLexer.TEST_STRUCT_TOKEN_TYPES)

    def test_token_symbols(self):
        data = """
        + - * / % | & ~ ^ << >>  ||  &&  !  <  <=  > >=  ==  !=
        =  *=  /=  %=  +=  -=  <<=  >>=  &=  ^=  |=
        ++ --
        ->
        ?
        () [] {} , . ; :
        ...
        """
        self.compare_token_output(data, expected_token_types=TestLexer.TEST_SYMBOLS_TOKEN_TYPES)

    def test_reserved_words(self):
        data = """
        auto
        break
        case
        char
        const
        continue
        default
        do
        double
        else
        enum
        extern
        float
        for
        goto
        if
        int
        long
        register
        return
        short
        signed
        sizeof
        static
        struct
        switch
        typedef
        union
        unsigned
        void
        volatile
        while
        """
        self.compare_token_output(data, expected_token_types=TestLexer.TEST_RESERVED_TOKEN_TYPES)

    def test_declare_function_pointer_typedef(self):
        data = """
        typedef int (*add_callback)(int a, int b);

        int add_two(int a, int b, add_callback callback);

        int normal_add(int a, int b);
        int weird_add(int a, int b);

        int main() {
          int x;
          int y;

          x = add_two(1, 2, normal_add);
          y = add_two(1, 2, weird_add);

          return 0;
        }

        int add_two(int a, int b, add_callback callback) {
            return callback(a, b);
        }

        int normal_add(int a, int b) {
            return a + b;
        }

        int weird_add(int a, int b) {
            return (a + b) % 4;
        }
        """
        self.compare_token_output(data, expected_token_types=TestLexer.TEST_FPTR_TOKEN_TYPES)

    def test_escaped_chars(self):
        data = """
            '\n' '\t' '\r'
        """

        self.compare_token_output(data, expected_token_types=['CCONST'] * 3)



    def test_int_verify_no_overflow(self):
        self.assertFalse(JSTLexer.string_to_int_fails("4"), "4 should be acceptable")

    def test_int_verify_overflow(self):
        self.assertTrue(JSTLexer.string_to_int_fails("9999999999999999999999999999999999999999"),
                        "That should should overflow")

    def test_float_acceptable(self):
        self.assertFalse(JSTLexer.string_to_float_fails('1.123'), "1.123 is an acceptable float")

    def test_float_unacceptable(self):
        self.assertTrue(JSTLexer.string_to_float_fails('1.8E+308'), "'1.8E+308' is too big")

    def compare_token_output(self, data, expected_token_types):
        self.source_code = data
        self.source_lines = data.split('\n')

        self.scanner.input(data)

        for given, expected in itertools.zip_longest(self.scanner, expected_token_types):
            self.assertEqual(given.type, expected)

if __name__ == '__main__':
    unittest.main()
