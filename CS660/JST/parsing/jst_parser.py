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
# File Description: The massive Parser file containing the productions and
# operations for parsing the ANSI C grammar.
###############################################################################

import compiler
from ast.ast_nodes import *
from exceptions.compile_error import CompileError
from scanning.jst_lexer import JSTLexer
from symbol_table.scope import Scope
from symbol_table.symbol import Symbol, VariableSymbol, FunctionSymbol, TypeDeclaration

import mips.library_functions as library_functions


# Parser Class
#
# This class is responsible for working in tandem with the Lexer to parse the given C program input and then
# constructing the Abstract Syntax Tree that corresponds to the program. Compile time checking is done by this class.
#
class JSTParser(object):

    # IMPORTANT: Tokens must be imported from the JSTLexer so PLY can build the table
    tokens = JSTLexer.tokens

    # The constructor for a new Parser object.
    #
    # @param self The object pointer.
    # @param compiler_state The object shared between the Parser and Lexer for maintaining the state of the compiler.
    #
    # Outputs: A constructed instance of a JSTParser.
    #
    # Purpose: The constructor initializes the object to be ready to do its job with the desired outputs labeled.
    def __init__(self, compiler_state):
        if compiler_state is None or not isinstance(compiler_state, compiler.compiler_state.CompilerState):
            raise ValueError('The passed compiler_state is not valid.')

        self.compiler_state = compiler_state
        self.prod_logger = self.compiler_state.get_parser_logger()
        self.warn_logger = self.compiler_state.get_warning_logger()

    # A method to do any cleanup that can't be handled by typical garbage collection.
    #
    # @param self The object pointer.
    #
    # Output: None
    #
    # Called by method responsible for completing the use of the parser cleanly.
    # TODO: implement __exit__ method?
    def teardown(self):
        self.prod_logger.finalize()

    ## Operator precedences used by ply.yacc to correctly order productions that may be otherwise ambiguous.
    #
    precedence = (
            ('left', 'LOR'),
            ('left', 'LAND'),
            ('left', 'OR'),
            ('left', 'XOR'),
            ('left', 'AND'),
            ('left', 'EQ', 'NE'),
            ('left', 'GT', 'GE', 'LT', 'LE'),
            ('left', 'RSHIFT', 'LSHIFT'),
            ('left', 'PLUS', 'MINUS'),
            ('left', 'TIMES', 'DIVIDE', 'MOD')
        )

    ## From here on productions will not be documented.
    #
    # It is generally expected that the reader understands the structure of a production handling function for ply.yacc,
    # where the production(s) handled are specified in the method's docstring.
    # In general, all of the methods function in the following way.
    #
    # @param self The object pointer.
    # @param t The production object assembled by ply.yacc, used to determine which production was taken, what the
    #          values of the elements of the production are, etc.
    #
    # Output:
    #   (via the t[0] item) Any result or AST node of a production.
    #
    # These methods are strictly called only by the parser itself.
    #

    #
    # p_error
    #
    def p_error(self, t):
        result = self.compiler_state.get_line_col_source(t.lineno, t.lexpos)
        raise CompileError('Parse failure.', result[0], result[1], result[2])

    def p_program(self, t):
        """
        program : enter_scope setup_for_program translation_unit_opt leave_scope
        """
        t[0] = t[3]

    #
    # translation-unit:
    #
    def p_translation_unit(self, t):
        """
        translation_unit_opt : translation_unit
                             | empty
        """
        self.output_production(t, production_message='translation_unit_opt -> translation_unit')

        t[0] = FileAST(t[1], self.compiler_state, lineno=None)

    def p_translation_unit_1(self, t):
        """
        translation_unit : external_declaration
        """
        self.output_production(t, production_message='translation_unit -> external_declaration')

        t[0] = t[1]

    def p_translation_unit_2(self, t):
        """
        translation_unit : translation_unit external_declaration
        """
        self.output_production(t, production_message='translation_unit -> translation_unit external_declaration')

        t[1].extend(t[2])
        t[0] = t[1]

    #
    # external-declaration:
    #
    def p_external_declaration_1(self, t):
        """
        external_declaration : function_definition
        """
        self.output_production(t, production_message='external_declaration -> function_definition')

        t[0] = [t[1]]

    def p_external_declaration_2(self, t):
        """
        external_declaration : declaration
        """
        self.output_production(t, production_message='external_declaration -> declaration')

        t[0] = []
        t[0].extend(t[1])

    #
    # function-definition
    #
    def p_function_definition_1(self, t):
        """
        function_definition : declarator enter_function_scope compound_statement leave_scope
        """
        self.output_production(t, production_message='function_definition -> declarator compound_statement')

        function_symbol = FunctionSymbol(t[1]['identifier'], t[1]['linecol'][0], t[1]['linecol'][1])
        function_symbol.set_named_parameters(t[1]['parameters'])       # TODO Setting parameters is required so that the function memory offset works correctly

        result, existing = self.compiler_state.symbol_table.insert(function_symbol)
        if result is Scope.INSERT_REDECL:
            if existing[0].finalized:
                tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
                raise CompileError.from_tuple('Reimplementation of function not allowed.', tup)
            else:
                existing[0].set_named_parameters(t[1]['parameters'])
                function_symbol = existing[0]

        type_declaration = TypeDeclaration()
        type_declaration.add_type_specifier('int')
        function_symbol.add_return_type_declaration(type_declaration)
        function_symbol.finalized = True

        # Manually set the frame size and clear counter
        function_symbol.activation_frame_size = self.compiler_state.symbol_table.next_activation_frame_offset
        self.compiler_state.symbol_table.next_activation_frame_offset = 0

        # Check if function is a definition of main
        if function_symbol.identifier == 'main':
            self.compiler_state.main_function = function_symbol

        arguments = function_symbol.named_parameters
        t[0] = FunctionDefinition(function_symbol, function_symbol.identifier, arguments, t[3],
                                  linerange=(t.lineno(1), t.lineno(4)))

    def p_function_definition_2(self, t):
        """
        function_definition : declaration_specifiers declarator enter_function_scope compound_statement leave_scope
        """
        self.output_production(t, production_message='function_definition -> declaration_specifiers declarator compound_statement')

        is_type_valid, message = type_utils.is_valid_type(t[1])
        if not is_type_valid:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple(message, tup)

        function_symbol = FunctionSymbol(t[2]['identifier'], t[2]['linecol'][0], t[2]['linecol'][1])
        function_symbol.set_named_parameters(t[2]['parameters'])       # TODO Setting parameters is required so that the function memory offset works correctly

        result, existing = self.compiler_state.symbol_table.insert(function_symbol)
        if result is Scope.INSERT_REDECL:
            if existing[0].finalized:
                tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
                raise CompileError.from_tuple('Reimplementation of function not allowed.', tup)
            else:
                existing[0].set_named_parameters(t[2]['parameters'])
                function_symbol = existing[0]

        function_symbol.add_return_type_declaration(t[1])
        function_symbol.finalized = True

        # Manually set the frame size and clear counter
        function_symbol.activation_frame_size = self.compiler_state.symbol_table.next_activation_frame_offset
        self.compiler_state.symbol_table.next_activation_frame_offset = 0

        # Check if function is a definition of main
        if function_symbol.identifier == 'main':
            self.compiler_state.main_function = function_symbol

        arguments = function_symbol.named_parameters
        t[0] = FunctionDefinition(function_symbol, function_symbol.identifier, arguments, t[4],
                                  linerange=(t.lineno(1), t.lineno(5)))

    def p_function_definition_3(self, t):
        """
        function_definition : declaration_specifiers declarator enter_function_scope declaration_list compound_statement leave_scope
        """
        self.output_production(t, production_message=
            'function_definition -> declaration_specifiers declarator declaration_list compound_statement')

        raise NotImplementedError('Unsupported form of function definition')

    def p_function_definition_4(self, t):
        """
        function_definition : declarator enter_function_scope declaration_list compound_statement leave_scope
        """
        self.output_production(t, production_message='function_definition -> declarator declaration_list compound_statement')

        raise NotImplementedError('Unsupported form of function definition')

    #
    # declaration:
    #
    def p_declaration_1(self, t):
        """
        declaration : declaration_specifiers init_declarator_list SEMI
        """
        self.output_production(t, production_message='declaration -> declaration_specifiers init_declarator_list SEMI')

        is_type_valid, message = type_utils.is_valid_type(t[1])
        if not is_type_valid:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple(message, tup)

        t[0] = []
        for init_declarator in t[2]:

            identifier = init_declarator['identifier']
            linecol = init_declarator['linecol']
            initializer = init_declarator['initializer']

            if 'parameters' in init_declarator:
                parameters = init_declarator['parameters']
                symbol = FunctionSymbol(identifier, linecol[0], linecol[1])
                symbol.add_return_type_declaration(t[1])
                symbol.set_named_parameters(parameters)

                ast_node = FunctionDeclaration(symbol, parameters, linerange=linecol)

            else:
                symbol = VariableSymbol(identifier, linecol[0], linecol[1])
                symbol.add_type_declaration(t[1])

                if 'array_dims' in init_declarator:
                    symbol.set_array_dims(init_declarator['array_dims'])

                if 'pointer_dims' in init_declarator:
                    symbol.set_pointer_dims(init_declarator['pointer_dims'])

                if symbol.immutable and isinstance(initializer, Constant):
                    symbol.value = initializer.value

                ast_node = Declaration(symbol, initializer, linerange=linecol)

            # Attempt to insert newly created symbol into the table
            result, _ = self.compiler_state.symbol_table.insert(symbol)
            if result is Scope.INSERT_REDECL:
                raise CompileError('Redeclaration of variable!', linecol[0], linecol[1],
                                   self.compiler_state.source_lines[linecol[0] - 1])
            elif result is Scope.INSERT_SHADOWED:
                if isinstance(symbol, FunctionSymbol):
                    raise CompileError('Redeclaration of function!', linecol[0], linecol[1],
                                       self.compiler_state.source_lines[linecol[0] - 1])
                else:
                    self.warn_logger.warning(str(
                        CompileError('Redeclaration of variable!', linecol[0], linecol[1],
                                     self.compiler_state.source_lines[linecol[0] - 1])))

            if ast_node:
                t[0].append(ast_node)
            else:
                raise Exception('Declaration AST is None!')

    def p_declaration_2(self, t):
        """
        declaration : declaration_specifiers SEMI
        """
        self.output_production(t, production_message='declaration -> declaration_specifiers SEMI')
        raise NotImplementedError('Unknown production.')

    #
    # declaration-list:
    #
    def p_declaration_list_1(self, t):
        """
        declaration_list : declaration
        """
        self.output_production(t, production_message='declaration_list -> declaration')

        t[0] = []
        t[0].extend(t[1])

    def p_declaration_list_2(self, t):
        """
        declaration_list : declaration_list declaration
        """
        self.output_production(t, production_message='declaration_list -> declaration_list declaration')

        t[1].extend(t[2])
        t[0] = t[1]

    #
    # declaration-specifiers
    #
    def p_declaration_specifiers_1(self, t):
        """
        declaration_specifiers : storage_class_specifier declaration_specifiers
        """
        self.output_production(t, production_message='declaration_specifiers -> storage_class_specifier declaration_specifiers')

        try:
            t[2].add_storage_class(t[1])
            t[0] = t[2]
        except Exception as ex:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError(str(ex), tup[0], tup[1], tup[2])

    def p_declaration_specifiers_2(self, t):
        """
        declaration_specifiers : type_specifier declaration_specifiers
        """
        self.output_production(t, production_message='declaration_specifiers -> type_specifier declaration_specifiers')

        try:
            t[2].add_type_specifier(t[1])
            t[0] = t[2]
        except Exception as ex:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError(str(ex), tup[0], tup[1], tup[2])

    def p_declaration_specifiers_3(self, t):
        """
        declaration_specifiers : type_qualifier declaration_specifiers
        """
        self.output_production(t, production_message='declaration_specifiers -> type_qualifier declaration_specifiers')

        try:
            t[2].add_type_qualifier(t[1])
            t[0] = t[2]
        except Exception as ex:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError(str(ex), tup[0], tup[1], tup[2])

    def p_declaration_specifiers_4(self, t):
        """
        declaration_specifiers : storage_class_specifier
        """
        self.output_production(t, production_message='declaration_specifiers -> storage_class_specifier')

        t[0] = TypeDeclaration()
        t[0].add_storage_class(t[1])

    def p_declaration_specifiers_5(self, t):
        """
        declaration_specifiers : type_specifier
        """
        self.output_production(t, production_message='declaration_specifiers -> type_specifier')

        t[0] = TypeDeclaration()
        t[0].add_type_specifier(t[1])

    def p_declaration_specifiers_6(self, t):
        """
        declaration_specifiers : type_qualifier
        """
        self.output_production(t, production_message='declaration_specifiers -> type_qualifier')

        t[0] = TypeDeclaration()
        t[0].add_type_qualifier(t[1])

    #
    # storage-class-specifier
    #
    def p_storage_class_specifier_auto(self, t):
        """storage_class_specifier : AUTO"""
        self.output_production(t, production_message='storage_class_specifier -> AUTO')

        t[0] = t[1]

    def p_storage_class_specifie_register(self, t):
        """storage_class_specifier : REGISTER"""
        self.output_production(t, production_message='storage_class_specifier -> REGISTER')

        t[0] = t[1]

    def p_storage_class_specifier_static(self, t):
        """storage_class_specifier : STATIC"""
        self.output_production(t, production_message='storage_class_specifier -> STATIC')

        t[0] = t[1]

    def p_storage_class_specifier_extern(self, t):
        """storage_class_specifier : EXTERN"""
        self.output_production(t, production_message='storage_class_specifier -> EXTERN')

        t[0] = t[1]

    def p_storage_class_specifier_typedef(self, t):
        """storage_class_specifier : TYPEDEF"""
        self.output_production(t, production_message='storage_class_specifier -> TYPEDEF')

        t[0] = t[1]

    #
    # type-specifier:
    #
    def p_type_specifier_void(self, t):
        """
        type_specifier : VOID
        """
        self.output_production(t, production_message='type_specifier -> VOID')

        t[0] = 'void'

    def p_type_specifier_char(self, t):
        """
        type_specifier : CHAR
        """
        self.output_production(t, production_message='type_specifier -> CHAR')

        t[0] = 'char'

    def p_type_specifier_short(self, t):
        """
        type_specifier : SHORT
        """
        self.output_production(t, production_message='type_specifier -> SHORT')

        t[0] = 'short'

    def p_type_specifier_int(self, t):
        """
        type_specifier : INT
        """
        self.output_production(t, production_message='type_specifier -> INT')

        t[0] = 'int'

    def p_type_specifier_long(self, t):
        """
        type_specifier : LONG
        """
        self.output_production(t, production_message='type_specifier -> LONG')

        t[0] = 'long'

    def p_type_specifier_float(self, t):
        """
        type_specifier : FLOAT
        """
        self.output_production(t, production_message='type_specifier -> FLOAT')

        t[0] = 'float'

    def p_type_specifier_double(self, t):
        """
        type_specifier : DOUBLE
        """
        self.output_production(t, production_message='type_specifier -> DOUBLE')

        t[0] = 'double'

    def p_type_specifier_signed(self, t):
        """
        type_specifier : SIGNED
        """
        self.output_production(t, production_message='type_specifier -> SIGNED')

        t[0] = 'signed'

    def p_type_specifier_unsigned(self, t):
        """
        type_specifier : UNSIGNED
        """
        self.output_production(t, production_message='type_specifier -> UNSIGNED')

        t[0] = 'unsigned'

    def p_type_specifier_struct_or_union(self, t):
        """
        type_specifier : struct_or_union_specifier
        """
        self.output_production(t, production_message='type_specifier -> struct_or_union_specifier')

        t[0] = t[1]

    def p_type_specifier_enum(self, t):
        """
        type_specifier : enum_specifier
        """
        self.output_production(t, production_message='type_specifier -> enum_specifier')

        t[0] = t[1]

    def p_type_specifier_typeid(self, t):
        """
        type_specifier : TYPEID
        """
        self.output_production(t, production_message='type_specifier -> TYPEID')

        t[0] = t[1]

    #
    # type-qualifier:
    #
    def p_type_qualifier_const(self, t):
        """type_qualifier : CONST"""
        self.output_production(t, production_message='type_qualifier -> CONST')

        t[0] = t[1]

    def p_type_qualifier_volatile(self, t):
        """type_qualifier : VOLATILE"""
        self.output_production(t, production_message='type_qualifier -> VOLATILE')

        t[0] = t[1]

    #
    # struct-or-union-specifier
    #
    def p_struct_or_union_specifier_1(self, t):
        """struct_or_union_specifier : struct_or_union identifier LBRACE struct_declaration_list RBRACE"""
        self.output_production(t, production_message=
            'struct_or_union_specifier : struct_or_union identifier LBRACE struct_declaration_list RBRACE')

        if t[1] is "struct":
            t[2].add_struct_members(t[4])
        elif t[1] is "union":
            t[2].add_union_members(t[4])

        t[0] = t[2]

    def p_struct_or_union_specifier_2(self, t):
        """
        struct_or_union_specifier : struct_or_union LBRACE struct_declaration_list RBRACE
        """
        self.output_production(t, production_message='struct_or_union_specifier : struct_or_union LBRACE struct_declaration_list RBRACE')

        raise NotImplemented('p_struct_or_union_specifier_2; how to do anonymous struct?')

    def p_struct_or_union_specifier_3(self, t):
        """
        struct_or_union_specifier : struct_or_union identifier
        """
        self.output_production(t, production_message='struct_or_union_specifier : struct_or_union identifier')

    #
    # struct-or-union:
    #
    def p_struct_or_union_struct(self, t):
        """struct_or_union : STRUCT"""
        self.output_production(t, production_message='struct_or_union -> STRUCT')

        t[0] = t[1]

    def p_struct_or_union_union(self, t):
        """struct_or_union : UNION"""
        self.output_production(t, production_message='struct_or_union -> UNION')

        t[0] = t[1]

    #
    # struct-declaration-list:
    #
    def p_struct_declaration_list_1(self, t):
        """struct_declaration_list : struct_declaration"""
        self.output_production(t, production_message='struct_declaration_list : struct_declaration')

        t[0] = [t[1]]

    def p_struct_declaration_list_2(self, t):
        """
        struct_declaration_list : struct_declaration_list struct_declaration
        """
        self.output_production(t, production_message='struct_declaration_list : struct_declaration_list struct_declaration')

        t[0] = t[1] + t[2]

    #
    # init-declarator-list:
    #
    def p_init_declarator_list_1(self, t):
        """
        init_declarator_list : init_declarator
        """
        self.output_production(t, production_message='init_declarator_list -> init_declarator')

        t[0] = [t[1]]

    def p_init_declarator_list_2(self, t):
        """
        init_declarator_list : init_declarator_list COMMA init_declarator
        """
        self.output_production(t, production_message='init_declarator_list -> init_declarator_list COMMA init_declarator')

        if 'parameters' in t[1][0] or 'parameters' in t[3]:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('Functions cannot be declared in a list.', tup)

        t[1].append(t[3])
        t[0] = t[1]

    #
    # init-declarator
    #
    def p_init_declarator_1(self, t):
        """
        init_declarator : declarator
        """
        self.output_production(t, production_message='init_declarator -> declarator')

        if 'array_dims' in t[1]:
            for dimension in t[1]['array_dims']:
                if dimension is None:
                    tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
                    raise CompileError.from_tuple('All dimensions in an array declaration must be'
                                                  ' specified if an initializer is not provided.', tup)

        t[1]['initializer'] = None
        t[0] = t[1]

    def p_init_declarator_2(self, t):
        """
        init_declarator : declarator EQUALS initializer
        """
        self.output_production(t, production_message='init_declarator -> declarator EQUALS initializer')

        if 'parameters' in t[1]:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('Functions cannot have initializers.', tup)

        if 'array_dims' in t[1]:
            for dimension in t[1]['array_dims'][1:]:
                if dimension is None:
                    tup = self.compiler_state.get_line_col_source(t.linepos(1), t.lexpos(1))
                    raise CompileError.from_tuple('Only the first dimension in an array declaration can be empty.', tup)

        if isinstance(t[3], list):
            if 'array_dims' not in t[1]:
                tup = self.compiler_state.get_line_col_source(t.lineno(3), t.lexpos(3))
                raise CompileError.from_tuple('Array initializer provided for non-array symbol.', tup)

            initializer_types = [type(item) for item in t[3]]

            if len(t[1]['array_dims']) is not 1 or (list in initializer_types):
                tup = self.compiler_state.get_line_col_source(t.lineno(3), t.lexpos(3))
                raise CompileError.from_tuple('Only 1D array initializers are allowed.', tup)

            # Fill in the empty dimension
            if t[1]['array_dims'][0] is None:
                t[1]['array_dims'][0] = len(t[3])

        else:
            if 'array_dims' in t[1]:
                tup = self.compiler_state.get_line_col_source(t.lineno(3), t.lexpos(3))
                raise CompileError.from_tuple('Non-array initializer provided for array symbol.', tup)

        t[1]['initializer'] = t[3]
        t[0] = t[1]

    #
    # struct-declaration:
    #
    def p_struct_declaration(self, t):
        """struct_declaration : specifier_qualifier_list struct_declarator_list SEMI"""
        self.output_production(t, production_message='struct_declaration -> specifier_qualifier_list struct_declarator_list SEMI')

    #
    # specifier-qualifier-list:
    #
    def p_specifier_qualifier_list_1(self, t):
        """specifier_qualifier_list : type_specifier specifier_qualifier_list"""
        self.output_production(t, production_message='specifier_qualifier_list -> type_specifier specifier_qualifier_list')

        t[2].add_all_from(t[1])
        t[0] = t[2]

    def p_specifier_qualifier_list_2(self, t):
        """specifier_qualifier_list : type_specifier"""
        self.output_production(t, production_message='specifier_qualifier_list -> type_specifier')

        type_declaration = TypeDeclaration()
        type_declaration.add_type_specifier(t[1])
        t[0] = type_declaration

    def p_specifier_qualifier_list_3(self, t):
        """specifier_qualifier_list : type_qualifier specifier_qualifier_list"""
        self.output_production(t, production_message='specifier_qualifier_list -> type_qualifier specifier_qualifier_list')

        t[2].add_all_from(t[1])
        t[0] = t[2]

    def p_specifier_qualifier_list_4(self, t):
        """specifier_qualifier_list : type_qualifier"""
        self.output_production(t, production_message='specifier_qualifier_list -> type_qualifier')

        type_declaration = TypeDeclaration()
        type_declaration.add_type_qualifier(t[1])
        t[0] = t[1]

    #
    # struct-declarator-list:
    #
    def p_struct_declarator_list_1(self, t):
        """struct_declarator_list : struct_declarator"""
        self.output_production(t, production_message='struct_declarator_list -> struct_declarator')

    def p_struct_declarator_list_2(self, t):
        """struct_declarator_list : struct_declarator_list COMMA struct_declarator"""
        self.output_production(t, production_message='struct_declarator_list -> struct_declarator_list COMMA struct_declarator')

    #
    # struct-declarator:
    #
    def p_struct_declarator_1(self, t):
        """struct_declarator : declarator"""
        self.output_production(t, production_message='struct_declarator -> declarator')

    def p_struct_declarator_2(self, t):
        """struct_declarator : declarator COLON constant_expression"""
        self.output_production(t, production_message='struct_declarator -> declarator COLON constant_expression')

    def p_struct_declarator_3(self, t):
        """struct_declarator : COLON constant_expression"""
        self.output_production(t, production_message='struct_declarator -> COLON constant_expression')

    #
    # enum-specifier:
    #
    def p_enum_specifier_1(self, t):
        """enum_specifier : ENUM identifier LBRACE enumerator_list RBRACE"""
        self.output_production(t, production_message='enum_specifier -> ENUM identifier LBRACE enumerator_list RBRACE')

        t[2].is_enum = True
        t[2].add_enum_members(t[4])
        t[0] = t[2]

    def p_enum_specifier_2(self, t):
        """enum_specifier : ENUM LBRACE enumerator_list RBRACE"""
        self.output_production(t, production_message='enum_specifier -> ENUM LBRACE enumerator_list RBRACE')

        raise NotImplemented('p_enum_specifier_2; how to do anonymous enum?')

    def p_enum_specifier_3(self, t):
        """enum_specifier : ENUM identifier"""
        self.output_production(t, production_message='enum_specifier -> ENUM identifier')

        t[2].is_enum = True
        t[0] = t[2]

    #
    # enumerator_list:
    #
    def p_enumerator_list_1(self, t):
        """enumerator_list : enumerator"""
        self.output_production(t, production_message='enumerator_list -> enumerator')

        t[0] = t[1]

    def p_enumerator_list_2(self, t):
        """enumerator_list : enumerator_list COMMA enumerator"""
        self.output_production(t, production_message='enumerator_list -> enumerator_list COMMA enumerator')

        t[0] = t[1].append_enum_member(t[3])

    #
    # enumerator:
    #
    def p_enumerator_1(self, t):
        """enumerator : identifier"""
        self.output_production(t, production_message='enumerator -> identifier')

        t[0] = t[1].set_constant_value(-1)  # TODO: create an enum value ticket counter

    def p_enumerator_2(self, t):
        """enumerator : identifier EQUALS constant_expression"""
        self.output_production(t, production_message='enumerator -> identifier EQUALS constant_expression')

        t[0] = t[1].set_constant_value(int(t[3][1]))  # t[3] should be a tuple like (value, type)
        # not checking for negativity, we can allow negative values for enums
        # TODO: create an enum value ticket counter

    #
    # declarator:
    #
    def p_declarator_1(self, t):
        """
        declarator : pointer direct_declarator
        """
        self.output_production(t, production_message='declarator -> pointer direct_declarator')

        if 'parameters' in t[2]:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('Is this a function pointer? Not supported.', tup)

        if 'pointer_dims' not in t[2]:
            t[2]['pointer_dims'] = t[1]
        else:
            t[2]['pointer_dims'].extend(t[1])

        # Don't forget to assign
        t[0] = t[2]

    def p_declarator_2(self, t):
        """
        declarator : direct_declarator
        """
        self.output_production(t, production_message='declarator -> direct_declarator')

        # Don't forget to assign
        t[0] = t[1]

    #
    # direct-declarator:
    #
    def p_direct_declarator_1(self, t):
        """
        direct_declarator : identifier
        """
        self.output_production(t, production_message='direct_declarator -> identifier')

        result = self.compiler_state.get_line_col(t, 1)
        t[0] = {'identifier': t[1], 'linecol': result}

    def p_direct_declarator_2(self, t):
        """
        direct_declarator : direct_declarator LPAREN RPAREN
        """
        self.output_production(t, production_message='direct_declarator -> direct_declarator LPAREN RPAREN')

        if 'parameters' in t[1]:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('Parameters have already been set.', tup)
        if 'array_dims' in t[1]:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('Functions cannot have dimensions.', tup)

        t[1]['parameters'] = []
        t[0] = t[1]

    def p_direct_declarator_3(self, t):
        """
        direct_declarator : direct_declarator LPAREN parameter_type_list RPAREN
        """
        self.output_production(t, production_message=
            'direct_declarator -> direct_declarator LPAREN parameter_type_list RPAREN')

        if 'parameters' in t[1]:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('Parameters have already been set.', tup)
        if 'array_dims' in t[1]:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('Functions cannot have dimensions.', tup)

        t[1]['parameters'] = t[3]
        t[0] = t[1]

    def p_direct_declarator_4(self, t):
        """
        direct_declarator : direct_declarator LBRACKET constant_expression_option RBRACKET
        """
        self.output_production(t, production_message=
            'direct_declarator -> direct_declarator LBRACKET constant_expression_option RBRACKET')

        if 'parameters' in t[1]:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('Functions cannot have dimensions.', tup)

        if 'array_dims' not in t[1]:
            t[1]['array_dims'] = []

        if t[3] is not None:
            if isinstance(t[3], VariableSymbol) and Constant.is_integral_type(t[3]) and t[3].immutable:
                t[1]['array_dims'].append(t[3].value)
            elif isinstance(t[3], Constant) and Constant.is_integral_type(t[3]):
                t[1]['array_dims'].append(t[3].value)
            else:
                tup = self.compiler_state.get_line_col_source(t.lineno(3), t.lexpos(3))
                raise CompileError.from_tuple('Non-constant value provided for array dimension.', tup)
        else:
            t[1]['array_dims'].append(VariableSymbol.EMPTY_ARRAY_DIM)

        # Don't forget to assign
        t[0] = t[1]

    def p_direct_declarator_5(self, t):
        """
        direct_declarator : LPAREN declarator RPAREN
        """
        self.output_production(t, production_message='direct_declarator -> LPAREN declarator RPAREN')
        raise NotImplementedError('Unknown production.')

    def p_direct_declarator_6(self, t):
        """
        direct_declarator : direct_declarator LPAREN identifier_list RPAREN
        """
        self.output_production(t, production_message=
            'direct_declarator -> direct_declarator LPAREN identifier_list RPAREN')
        raise NotImplementedError('Unknown production.')

    #
    # pointer:
    #
    def p_pointer_1(self, t):
        """
        pointer : TIMES type_qualifier_list
        """
        self.output_production(t, production_message='pointer -> TIMES type_qualifier_list')

        t[0] = [t[2]]

    def p_pointer_2(self, t):
        """
        pointer : TIMES
        """
        self.output_production(t, production_message='pointer -> TIMES')

        t[0] = [set()]

    def p_pointer_3(self, t):
        """
        pointer : TIMES type_qualifier_list pointer
        """
        self.output_production(t, production_message='pointer -> TIMES type_qualifier_list pointer')

        t[3].extend(t[2])
        t[0] = t[3]

    def p_pointer_4(self, t):
        """
        pointer : TIMES pointer
        """
        self.output_production(t, production_message='pointer -> TIMES pointer')

        t[2].extend([set()])
        t[0] = t[2]

    #
    # type-qualifier-list:
    #
    def p_type_qualifier_list_1(self, t):
        """
        type_qualifier_list : type_qualifier
        """
        self.output_production(t, production_message='type_qualifier_list -> type_qualifier')

        t[0] = set(t[1])

    def p_type_qualifier_list_2(self, t):
        """
        type_qualifier_list : type_qualifier_list type_qualifier
        """
        self.output_production(t, production_message='type_qualifier_list -> type_qualifier_list type_qualifier')

        t[1].add(t[2])
        t[0] = t[1]

    #
    # parameter-type-list:
    #
    def p_parameter_type_list_1(self, t):
        """parameter_type_list : parameter_list"""
        self.output_production(t, production_message='parameter_type_list -> parameter_list')

        t[0] = t[1]

    def p_parameter_type_list_2(self, t):
        """parameter_type_list : parameter_list COMMA ELLIPSIS"""
        self.output_production(t, production_message='parameter_type_list -> parameter_list COMMA ELLIPSIS')
        raise Exception('Ellipsis is not supported')

    #
    # parameter-list:
    #
    def p_parameter_list_1(self, t):
        """parameter_list : parameter_declaration"""
        self.output_production(t, production_message='parameter_list -> parameter_declaration')

        t[0] = [t[1]]

    def p_parameter_list_2(self, t):
        """parameter_list : parameter_list COMMA parameter_declaration"""
        self.output_production(t, production_message='parameter_list -> parameter_list COMMA parameter_declaration')

        t[0] = t[1] + [t[3]]

    #
    # parameter-declaration:
    #
    def p_parameter_declaration_1(self, t):
        """
        parameter_declaration : declaration_specifiers
        """
        self.output_production(t, production_message='parameter_declaration -> declaration_specifiers')

        is_type_valid, message = type_utils.is_valid_type(t[1])
        if not is_type_valid:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple(message, tup)

        tup = self.compiler_state.get_line_col(t, 1)
        symbol = VariableSymbol('', tup[0], tup[1])
        symbol.add_type_declaration(t[1])
        t[0] = symbol

    def p_parameter_declaration_2(self, t):
        """
        parameter_declaration : declaration_specifiers declarator
        """
        self.output_production(t, production_message='parameter_declaration -> declaration_specifiers declarator')

        is_type_valid, message = type_utils.is_valid_type(t[1])
        if not is_type_valid:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple(message, tup)

        if 'parameters' in t[2]:
            tup = self.compiler_state.get_line_col_source(t.lineno(2), t.lexpos(2))
            raise CompileError.from_tuple(message, tup)

        identifier = t[2]['identifier']
        linecol = t[2]['linecol']
        symbol = VariableSymbol(identifier, linecol[0], linecol[1])
        symbol.add_type_declaration(t[1])

        if 'array_dims' in t[2]:
            symbol.set_array_dims(t[2]['array_dims'])

        if 'pointer_dims' in t[2]:
            symbol.set_pointer_dims(t[2]['pointer_dims'])

        # Don't forget to assign
        t[0] = symbol

    def p_parameter_declaration_3(self, t):
        """
        parameter_declaration : declaration_specifiers abstract_declarator
        """
        self.output_production(t, production_message='parameter_declaration -> declaration_specifiers abstract_declarator')

        raise NotImplementedError('parameter_declaration : declaration_specifiers abstract_declarator')

    #
    # identifier-list:
    #
    def p_identifier_list_1(self, t):
        """
        identifier_list : identifier
        """
        self.output_production(t, production_message='identifier_list -> identifier')

        t[0] = [t[1]]

    def p_identifier_list_2(self, t):
        """
        identifier_list : identifier_list COMMA identifier
        """
        self.output_production(t, production_message='identifier_list -> identifier_list COMMA identifier')

        t[0] = t[1].append(t[3])

    #
    # initializer:
    #
    def p_initializer_1(self, t):
        """
        initializer : assignment_expression
        """
        self.output_production(t, production_message='initializer -> assignment_expression')

        # If the initializer is a string
        if isinstance(t[1], str):
            t[0] = [x for x in t[1]]
            t[0][-1] = '\0'
            t[0] = t[0][1:]
        else:
            t[0] = t[1]

    def p_initializer_2(self, t):
        """
        initializer : LBRACE initializer_list RBRACE
        """
        self.output_production(t, production_message='initializer -> LBRACE initializer_list RBRACE')

        t[0] = t[2]

    def p_initializer_3(self, t):
        """
        initializer : LBRACE initializer_list COMMA RBRACE
        """
        self.output_production(t, production_message='initializer -> LBRACE initializer_list COMMA RBRACE')

        t[0] = t[2]

    #
    # initializer-list:
    #
    def p_initializer_list_1(self, t):
        """
        initializer_list : initializer
        """
        self.output_production(t, production_message='initializer_list -> initializer')

        t[0] = [t[1]]

    def p_initializer_list_2(self, t):
        """
        initializer_list : initializer_list COMMA initializer
        """
        self.output_production(t, production_message='initializer_list -> initializer_list COMMA initializer')

        t[1].append(t[3])
        t[0] = t[1]

    #
    # type-name:
    #
    def p_type_name_1(self, t):
        """
        type_name : specifier_qualifier_list abstract_declarator
        """
        self.output_production(t, production_message='type_name -> specifier_qualifier_list abstract_declarator')
        raise NotImplemented()

    def p_type_name_2(self, t):
        """
        type_name : specifier_qualifier_list
        """
        self.output_production(t, production_message='type_name -> specifier_qualifier_list')

        t[0] = t[1].get_type_str()

    #
    # abstract-declarator:
    #
    def p_abstract_declarator_1(self, t):
        """abstract_declarator : pointer"""
        self.output_production(t, production_message='abstract_declarator -> pointer')

        t[0] = {'pointer_modifiers': t[1]}

    def p_abstract_declarator_2(self, t):
        """abstract_declarator : pointer direct_abstract_declarator"""
        self.output_production(t, production_message='abstract_declarator -> pointer direct_abstract_declarator')

        t[2].update({'pointer_modifiers': t[1]})

        t[0] = t[2]


    def p_abstract_declarator_3(self, t):
        """abstract_declarator : direct_abstract_declarator"""
        self.output_production(t, production_message='abstract_declarator -> direct_abstract_declarator')

        t[0] = t[1]

    #
    # direct-abstract-declarator:
    #
    def p_direct_abstract_declarator_1(self, t):
        """direct_abstract_declarator : LPAREN abstract_declarator RPAREN"""
        self.output_production(t, production_message='direct_abstract_declarator -> LPAREN abstract_declarator RPAREN')

        t[0] = t[1]
        raise NotImplementedError('Unknown production.')

    def p_direct_abstract_declarator_2(self, t):
        """direct_abstract_declarator : direct_abstract_declarator LBRACKET constant_expression_option RBRACKET"""
        self.output_production(t, production_message=
            'direct_abstract_declarator -> direct_abstract_declarator LBRACKET constant_expression_option RBRACKET')

        if 'array_dims' not in t[1]:
            t[1]['array_dims'] = []

        t[1]['array_dims'] += [t[3]]
        t[0] = t[1]

    def p_direct_abstract_declarator_3(self, t):
        """direct_abstract_declarator : LBRACKET constant_expression_option RBRACKET"""
        self.output_production(t, production_message='direct_abstract_declarator -> LBRACKET constant_expression_option RBRACKET')

        t[0] = {'array_dims': [t[2]]}

    def p_direct_abstract_declarator_4(self, t):
        """direct_abstract_declarator : direct_abstract_declarator LPAREN parameter_type_list_option RPAREN"""
        self.output_production(t, production_message=
            'direct_abstract_declarator -> direct_abstract_declarator LPAREN parameter_type_list_option RPAREN')

        raise NotImplementedError('Unknown production. Possibly for function pointers.')

    def p_direct_abstract_declarator_5(self, t):
        """
        direct_abstract_declarator : LPAREN parameter_type_list_option RPAREN
        """
        self.output_production(t, production_message='direct_abstract_declarator -> LPAREN parameter_type_list_option RPAREN')

        raise NotImplementedError('Unknown production. Possibly for function pointers.')

    #
    # Optional fields in abstract declarators
    #
    def p_constant_expression_option_to_empty(self, t):
        """
        constant_expression_option : empty
        """
        self.output_production(t, production_message='constant_expression_option -> empty')

        t[0] = None

    def p_constant_expression_option_to_constant_expression(self, t):
        """
        constant_expression_option : constant_expression
        """
        self.output_production(t, production_message='constant_expression_option -> constant_expression')

        t[0] = t[1]

    def p_parameter_type_list_option_to_empty(self, t):
        """
        parameter_type_list_option : empty
        """
        self.output_production(t, production_message='parameter_type_list_option -> empty')

    def p_parameter_type_list_option_to_parameter_type_list(self, t):
        """
        parameter_type_list_option : parameter_type_list
        """
        self.output_production(t, production_message='parameter_type_list_option -> parameter_type_list')

    #
    # statement:
    #
    def p_statement_labeled(self, t):
        """
        statement : labeled_statement
        """
        self.output_production(t, production_message='statement -> labeled_statement')

        t[0] = t[1]

    def p_statement_expression(self, t):
        """
        statement : expression_statement
        """
        self.output_production(t, production_message='statement -> expression_statment')

        t[0] = t[1]

    def p_statement_to_compound_statement(self, t):
        """
        statement : compound_statement
        """
        self.output_production(t, production_message='statement -> compound_statement')

        t[0] = t[1]

    def p_statement_selection(self, t):
        """
        statement : selection_statement
        """
        self.output_production(t, production_message='statement -> selection_statement')

        t[0] = t[1]

    def p_statement_iteration(self, t):
        """
        statement : iteration_statement
        """
        self.output_production(t, production_message='statement -> iteration_statement')

        t[0] = t[1]

    def p_statement_jump(self, t):
        """
        statement : jump_statement
        """
        self.output_production(t, production_message='statement -> jump_statement')

        t[0] = t[1]

    #
    # labeled-statement:
    #
    def p_labeled_statement_1(self, t):
        """
        labeled_statement : identifier COLON statement
        """
        self.output_production(t, production_message='labeled_statement -> identifier COLON statement')

    def p_labeled_statement_2(self, t):
        """
        labeled_statement : CASE constant_expression COLON statement
        """
        self.output_production(t, production_message='labeled_statement -> CASE constant_expression COLON statement')

    def p_labeled_statement_3(self, t):
        """
        labeled_statement : DEFAULT COLON statement
        """
        self.output_production(t, production_message='labeled_statement -> DEFAULT COLON statement')

    #
    # expression-statement:
    #
    def p_expression_statement(self, t):
        """
        expression_statement : expression_option SEMI
        """
        self.output_production(t, production_message='expression_statement -> expression_option SEMI')

        t[0] = t[1]

    #
    # compound-statement:
    #
    def p_compound_statement_1(self, t):
        """
        compound_statement : LBRACE enter_scope insert_mode declaration_list lookup_mode statement_list leave_scope RBRACE
        """
        self.output_production(t, production_message='compound_statement -> LBRACE declaration_list statement_list RBRACE')

        t[0] = CompoundStatement(declaration_list=t[4], statement_list=t[6], linerange=(t.lineno(1), t.lineno(8)))

    def p_compound_statement_2(self, t):
        """
        compound_statement : LBRACE enter_scope lookup_mode statement_list leave_scope RBRACE
        """
        self.output_production(t, production_message='compound_statement -> LBRACE statement_list RBRACE')

        t[0] = CompoundStatement(statement_list=t[4], linerange=(t.lineno(1), t.lineno(6)))

    def p_compound_statement_3(self, t):
        """
        compound_statement : LBRACE enter_scope insert_mode declaration_list lookup_mode leave_scope RBRACE
        """
        self.output_production(t, production_message='compound_statement -> LBRACE declaration_list RBRACE')

        t[0] = CompoundStatement(declaration_list=t[4], linerange=(t.lineno(1), t.lineno(7)))

    def p_compound_statement_4(self, t):
        """
        compound_statement : LBRACE RBRACE
        """
        self.output_production(t, production_message='compound_statement -> LBRACE RBRACE')

        t[0] = None

    #
    # statement-list:
    #
    def p_statement_list_1(self, t):
        """
        statement_list : statement
        """
        self.output_production(t, production_message='statement_list -> statement')

        t[0] = [t[1]]

    def p_statement_list_2(self, t):
        """
        statement_list : statement_list statement
        """
        self.output_production(t, production_message='statement_list -> statement_list statement')

        t[1].append(t[2])
        t[0] = t[1]

    #
    # selection-statement
    #
    def p_selection_statement_1(self, t):
        """
        selection_statement : IF LPAREN expression RPAREN statement
        """
        self.output_production(t, production_message='selection_statement -> IF LPAREN expression RPAREN statement')

        t[0] = If(conditional=t[3], if_true=t[5], if_false=None, linerange=(t.lineno(1), t.lineno(5)))

    def p_selection_statement_2(self, t):
        """
        selection_statement : IF LPAREN expression RPAREN statement ELSE statement
        """
        self.output_production(t,
            production_message='selection_statement -> IF LPAREN expression RPAREN statement ELSE statement')

        t[0] = If(conditional=t[3], if_true=t[5], if_false=t[7], linerange=(t.lineno(1), t.lineno(7)))

    def p_selection_statement_3(self, t):
        """
        selection_statement : SWITCH LPAREN expression RPAREN statement
        """
        self.output_production(t, production_message='selection_statement -> SWITCH LPAREN expression RPAREN statement')

    #
    # iteration_statement:
    #
    def p_iteration_statement_1(self, t):
        """
        iteration_statement : WHILE LPAREN expression RPAREN statement
        """
        self.output_production(t, production_message='iteration_statement -> WHILE LPAREN expression RPAREN statement')

        t[0] = IterationNode(True, None, t[3], None, t[5], linerange=(t.lineno(1), t.lineno(4)))

    def p_iteration_statement_2(self, t):
        """
        iteration_statement : FOR LPAREN expression_option SEMI expression_option SEMI expression_option RPAREN statement
        """
        self.output_production(t, production_message=
            'iteration_statement -> FOR LPAREN expression_option SEMI expression_option SEMI expression_option RPAREN '
            'statement')

        t[0] = IterationNode(True, t[3], t[5], t[7], t[9], linerange=(t.lineno(1), t.lineno(8)))

    def p_iteration_statement_3(self, t):
        """
        iteration_statement : DO statement WHILE LPAREN expression RPAREN SEMI
        """
        self.output_production(t, production_message='iteration_statement -> DO statement WHILE LPAREN expression RPAREN SEMI')

        t[0] = IterationNode(False, None, t[5], None, t[2], linerange=(t.lineno(1), t.lineno(7)))

    #
    # jump_statement:
    #
    def p_jump_statement_1(self, t):
        """
        jump_statement : GOTO identifier SEMI
        """
        self.output_production(t, production_message='jump_statement -> GOTO identifier SEMI')

    def p_jump_statement_2(self, t):
        """
        jump_statement : CONTINUE SEMI
        """
        self.output_production(t, production_message='jump_statement -> CONTINUE SEMI')

    def p_jump_statement_3(self, t):
        """
        jump_statement : BREAK SEMI
        """
        self.output_production(t, production_message='jump_statement -> BREAK SEMI')

    def p_jump_statement_4(self, t):
        """
        jump_statement : RETURN expression_option SEMI
        """
        self.output_production(t, production_message='jump_statement -> RETURN expression_option SEMI')

        t[0] = Return(expression=t[2] if t[2] else None, linerange=(t.lineno(1), t.lineno(3)))

    #
    # Expression Option
    #
    def p_expression_option_1(self, t):
        """
        expression_option : empty
        """
        self.output_production(t, production_message='expression_option -> empty')

    def p_expression_option_2(self, t):
        """
        expression_option : expression
        """
        self.output_production(t, production_message='expression_option -> expression')

        t[0] = t[1]

    #
    # expression:
    #
    def p_expression_1(self, t):
        """
        expression : assignment_expression
        """
        self.output_production(t, production_message='expression -> assignment_expression')

        t[0] = t[1]

    def p_expression_2(self, t):
        """
        expression : expression COMMA assignment_expression
        """
        self.output_production(t, production_message='expression -> expression COMMA assignment_expression')
        raise NotImplemented('Build a (list of Expression) here')

    #
    # assigment_expression:
    #
    def p_assignment_expression_pass_through(self, t):
        """
        assignment_expression : conditional_expression
        """
        self.output_production(t, production_message='assignment_expression -> conditional_expression')

        t[0] = t[1]

    def p_assignment_expression_2(self, t):
        """
        assignment_expression : unary_expression assignment_operator assignment_expression
        """
        self.output_production(t, production_message=
            'assignment_expression -> unary_expression assignment_operator assignment_expression')

        if t[1].immutable:
            line, column, source_code = self.compiler_state.get_line_col_source(t.lineno(2), t.lexpos(2))
            raise CompileError("Assignment to immutable types is not allowed", line, column, source_code)

        if isinstance(t[1], FunctionSymbol):
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('The assignment operator cannot be applied to functions.', tup)

        if isinstance(t[1], ArrayReference):  # LEFT side
            valid, message = t[1].check_subscripts()
            if not valid:
                tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
                raise CompileError.from_tuple(message, tup)

        if isinstance(t[3], ArrayReference):  # RIGHT side
            valid, message = t[3].check_subscripts()
            if not valid:
                tup = self.compiler_state.get_line_col_source(t.lineno(3), t.lexpos(3))
                raise CompileError.from_tuple(message, tup)

        cast_result, message = type_utils.can_assign(t[1].get_resulting_type(), t[3].get_resulting_type())
        if cast_result != type_utils.INCOMPATIBLE_TYPES:

            if t[2] == '=':
                t[0] = Assignment(t[2], t[1], t[3], linerange=(t.lineno(1), t.lineno(3)))

            else:
                operator = t[2].replace('=', '')
                binary_operation = BinaryOperator(operator, t[1], t[3], linerange=(t.lineno(1), t.lineno(3)))
                t[0] = Assignment(t[2], t[1], binary_operation, linerange=(t.lineno(1), t.lineno(3)))

        else:
            tup = self.compiler_state.get_line_col_source(t.lineno(3), t.lexpos(3))
            raise CompileError(message, tup[0], tup[1], tup[2])

    #
    # assignment_operator:
    #
    def p_assignment_operator(self, t):
        """
        assignment_operator : EQUALS
                            | XOREQUAL
                            | TIMESEQUAL
                            | DIVEQUAL
                            | MODEQUAL
                            | PLUSEQUAL
                            | MINUSEQUAL
                            | LSHIFTEQUAL
                            | RSHIFTEQUAL
                            | ANDEQUAL
                            | OREQUAL
        """
        self.output_production(t, production_message='assignment_operator -> {}'.format(t[1]))

        t[0] = t[1]

    #
    # constant-expression
    #
    def p_constant_expression(self, t):
        """
        constant_expression : conditional_expression
        """
        self.output_production(t, production_message='constant_expression -> conditional_expression')

        t[0] = t[1]

    #
    # conditional-expression
    #
    def p_conditional_expression_to_binary_expression(self, t):
        """
        conditional_expression : binary_expression
        """
        self.output_production(t, production_message='conditional_expression -> binary_expression')

        t[0] = t[1]

    def p_conditional_expression_to_ternary_expression(self, t):
        """
        conditional_expression : binary_expression CONDOP expression COLON conditional_expression
        """
        self.output_production(t, production_message=
            'conditional_expression -> binary_expression CONDOP expression COLON conditional_expression')

        raise NotImplementedError('Ternary operator')

    #
    # binary-expression
    #
    def p_binary_expression_to_implementation(self, t):
        """
        binary_expression : binary_expression TIMES binary_expression
                          | binary_expression DIVIDE binary_expression
                          | binary_expression MOD binary_expression
                          | binary_expression PLUS binary_expression
                          | binary_expression MINUS binary_expression
                          | binary_expression RSHIFT binary_expression
                          | binary_expression LSHIFT binary_expression
                          | binary_expression LT binary_expression
                          | binary_expression LE binary_expression
                          | binary_expression GE binary_expression
                          | binary_expression GT binary_expression
                          | binary_expression EQ binary_expression
                          | binary_expression NE binary_expression
                          | binary_expression AND binary_expression
                          | binary_expression OR binary_expression
                          | binary_expression XOR binary_expression
                          | binary_expression LAND binary_expression
                          | binary_expression LOR binary_expression
        """
        self.output_production(t, production_message=
            'binary_expression -> binary_expression {} binary_expression'.format(t[2]))

        # If constant folding is possible
        if JSTParser.compile_time_evaluable(t[1]) and JSTParser.compile_time_evaluable(t[3]):
            t[0] = JSTParser.perform_binary_operation(t[1], t[2], t[3])

        # Else, binary operator node needed
        else:
            t[0] = BinaryOperator(t[2], t[1], t[3], linerange=(t.lineno(1), t.lineno(3)))

    def p_binary_expression_to_cast_expression(self, t):
        """
        binary_expression : cast_expression
        """
        self.output_production(t, production_message='binary_expression -> cast_expression')

        t[0] = t[1]

    #
    # cast_expression:
    #
    def p_cast_expression_1(self, t):
        """
        cast_expression : unary_expression
        """
        self.output_production(t, production_message='cast_expression -> unary_expression')

        t[0] = t[1]

    def p_cast_expression_2(self, t):
        """
        cast_expression : LPAREN type_name RPAREN cast_expression
        """
        self.output_production(t, production_message='cast_expression -> LPAREN type_name RPAREN cast_expression')

        t[0] = Cast(t[2], t[4], linerange=(t.lineno(1), t.lineno(4)))

    #
    # unary_expression:
    #
    def p_unary_expression_to_postfix_expression(self, t):
        """
        unary_expression : postfix_expression
        """
        self.output_production(t, production_message='unary_expression -> postfix_expression')

        t[0] = t[1]

    def p_unary_expression_pre_plus_plus(self, t):
        """
        unary_expression : PLUSPLUS unary_expression
        """
        self.output_production(t, production_message='unary_expression -> PLUSPLUS unary_expression')

        t[0] = UnaryOperator(operator=t[1], pre=True, expression=t[2], linerange=(t.lineno(1), t.lineno(2)))

    def p_unary_expression_pre_minus_minus(self, t):
        """
        unary_expression : MINUSMINUS unary_expression
        """
        self.output_production(t, production_message='unary_expression -> MINUSMINUS unary_expression')

        t[0] = UnaryOperator(operator=t[1], pre=True, expression=t[2], linerange=(t.lineno(1), t.lineno(2)))

    def p_unary_expression_to_unary_operator_and_cast(self, t):
        """
        unary_expression : unary_operator cast_expression
        """
        self.output_production(t, production_message='unary_expression -> unary_operator cast_expression')

        t[0] = t[2]  # TODO: handle appropriately with AST nodes

    def p_unary_expression_sizeof(self, t):
        """
        unary_expression : SIZEOF unary_expression
        """
        self.output_production(t, production_message='unary_expression -> SIZEOF unary_expression')
        raise NotImplemented()

    def p_unary_expression_sizeof_parenthesized(self, t):
        """
        unary_expression : SIZEOF LPAREN type_name RPAREN
        """
        self.output_production(t, production_message='unary_expression -> SIZEOF LPAREN type_name RPAREN')
        raise NotImplemented()

    #
    # unary_operator
    #
    def p_unary_operator_and(self, t):
        """
        unary_operator : AND
        """
        self.output_production(t, production_message='unary_operator -> AND')

    def p_unary_operator_times(self, t):
        """
        unary_operator : TIMES
        """
        self.output_production(t, production_message='unary_operator -> TIMES')

    def p_unary_operator_plus(self, t):
        """unary_operator : PLUS
        """
        self.output_production(t, production_message='unary_operator -> PLUS')

    def p_unary_operator_minus(self, t):
        """unary_operator : MINUS
        """
        self.output_production(t, production_message='unary_operator -> MINUS')

    def p_unary_operator_not(self, t):
        """unary_operator : NOT
        """
        self.output_production(t, production_message='unary_operator -> NOT')

    def p_unary_operator_lnot(self, t):
        """
        unary_operator : LNOT
        """
        self.output_production(t, production_message='unary_operator -> LNOT')

    #
    # postfix_expression:
    #
    def p_postfix_expression_to_primary_expression(self, t):
        """
        postfix_expression : primary_expression
        """
        self.output_production(t, production_message='postfix_expression -> primary_expression')

        t[0] = t[1]

    def p_postfix_expression_to_array_dereference(self, t):
        """
        postfix_expression : postfix_expression LBRACKET expression RBRACKET
        """
        self.output_production(t, production_message='postfix_expression -> postfix_expression LBRACKET expression RBRACKET')

        if isinstance(t[1], FunctionSymbol):
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('Functions cannot be accessed like arrays.', tup)

        if isinstance(t[1], VariableSymbol):

            if len(t[1].array_dims) > 0 or len(t[1].pointer_dims) > 0:
                # TODO Convert to VariableDereference to support pointer dereference? - Shubham (sg-variable-symbol)
                t[0] = ArrayReference(t[1], [t[3]], linerange=(t.lineno(1), t.lineno(4)))
                return
            else:
                tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
                raise CompileError.from_tuple('Symbol is not an array.', tup)

        elif isinstance(t[1], ArrayReference):

            if len(t[1].subscripts) < (len(t[1].symbol.array_dims) + len(t[1].symbol.pointer_dims)):
                t[1].subscripts.append(t[3])
                t[0] = t[1]
            else:
                tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
                raise CompileError.from_tuple('Too many subscripts. Symbol does not have that many dimensions.', tup)

        else:

            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('Unknown postfix expression {}'.format(type(t[1])), tup)

    def p_postfix_expression_to_parameterized_function_call(self, t):
        """
        postfix_expression : postfix_expression LPAREN argument_expression_list RPAREN
        """
        self.output_production(t, production_message='postfix_expression -> postfix_expression LPAREN argument_expression_list RPAREN')

        if isinstance(t[1], FunctionSymbol):
            matched, message = t[1].arguments_match_parameter_types(t[3])

            if matched:
                t[0] = FunctionCall(t[1], t[3], linerange=(t.lineno(1), t.lineno(4)))
            else:
                tup = self.compiler_state.get_line_col_source(t.lineno(3), t.lexpos(3))
                raise CompileError.from_tuple(message, tup)
        else:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('Unknown postfix_expression ({})'.format(type(t[1])), tup)

    def p_postfix_expression_to_function_call(self, t):
        """
        postfix_expression : postfix_expression LPAREN RPAREN
        """
        self.output_production(t, production_message='postfix_expression -> postfix_expression LPAREN RPAREN')

        if isinstance(t[1], FunctionSymbol):
            matched, message = t[1].arguments_match_parameter_types([])

            if matched:
                t[0] = FunctionCall(t[1], None, linerange=(t.lineno(1), t.lineno(3)))
            else:
                tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
                raise CompileError.from_tuple(message, tup)
        else:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('Unknown postfix_expression ({})'.format(type(t[1])), tup)

    def p_postfix_expression_to_struct_member_access(self, t):
        """
        postfix_expression : postfix_expression PERIOD identifier
        """
        self.output_production(t, production_message='postfix_expression -> postfix_expression PERIOD identifier')
        raise NotImplemented('Used for structs, unions')

    def p_postfix_expression_to_struct_member_dereference(self, t):
        """
        postfix_expression : postfix_expression ARROW identifier
        """
        self.output_production(t, production_message='postfix_expression -> postfix_expression ARROW identifier')
        raise NotImplemented('Used to dereference pointers and access member')

    def p_postfix_expression_to_post_increment(self, t):
        """
        postfix_expression : postfix_expression PLUSPLUS
        """
        self.output_production(t, production_message='postfix_expression -> postfix_expression PLUSPLUS')

        # Pre set to false will increment AFTER
        t[0] = UnaryOperator(operator=t[2], pre=False, expression=t[1], linerange=(t.lineno(1), t.lineno(2)))

    def p_postfix_expression_to_post_decrement(self, t):
        """
        postfix_expression : postfix_expression MINUSMINUS
        """
        self.output_production(t, production_message='postfix_expression -> postfix_expression MINUSMINUS')

        # Pre set to false will increment AFTER
        t[0] = UnaryOperator(operator=t[2], pre=False, expression=t[1], linerange=(t.lineno(1), t.lineno(2)))

    #
    # primary-expression:
    #
    def p_primary_expression_identifier(self, t):
        """
        primary_expression :  identifier
        """
        self.output_production(t, production_message='primary_expression -> identifier')

        symbol, _ = self.compiler_state.symbol_table.find(t[1])
        if symbol is None:
            tup = self.compiler_state.get_line_col_source(t.lineno(1), t.lexpos(1))
            raise CompileError.from_tuple('Use of variable before declaration.', tup)

        # Don't forget to assign
        t[0] = symbol

    def p_primary_expression_constant(self, t):
        """
        primary_expression : constant
        """
        self.output_production(t, production_message='primary_expression -> constant')

        t[0] = t[1]

    def p_primary_expression_string_literal(self, t):
        """
        primary_expression : string_literal
        """
        self.output_production(t, production_message='primary_expression -> string_literal_list')

        t[0] = t[1]

    def p_string_literal_fragment(self, t):
        """
        string_literal : SCONST
        """
        self.output_production(t, production_message='string_literal_list -> SCONST {}'.format(t[1]))

        t[0] = t[1]

    def p_string_literal_plus_string_literal_fragment(self, t):
        """
        string_literal : string_literal SCONST
        """
        self.output_production(t, production_message='string_literal_list -> string_literal_list SCONST')

        # concatenate the string fragments into a single string literal by trimming off the quote marks
        t[0] = t[1][:-1] + t[2][1:]

    def p_primary_expression_parenthesized(self, t):
        """
        primary_expression : LPAREN expression RPAREN
        """
        self.output_production(t, production_message='primary_expression -> LPAREN expression RPAREN')

        # a parenthesised expression evaluates to the expression itself
        t[0] = t[2]

    #
    # argument-expression-list:
    #
    def p_argument_expression_list_assignment_expression(self, t):
        """
        argument_expression_list : assignment_expression
        """
        self.output_production(t, production_message='argument_expression_list -> assignment_expression')

        t[0] = [t[1]]

    def p_argument_expression_list_list_comma_expression(self, t):
        """
        argument_expression_list : argument_expression_list COMMA assignment_expression
        """
        self.output_production(t, production_message=
            'argument_expression_list -> argument_expression_list COMMA assignment_expression')

        t[0] = t[1] + [t[3]]

    #
    # constant:
    #
    def p_constant_int(self, t):
        """
        constant : ICONST
        """
        self.output_production(t, production_message='constant -> ICONST {}'.format(t[1]))

        if t[1][1] is 'CHAR':
            t[0] = Constant(Constant.CHAR, t[1][0], linerange=(t.lineno(1), t.lineno(1)))

        elif t[1][1] is 'INT':
            t[0] = Constant(Constant.INTEGER, t[1][0], linerange=(t.lineno(1), t.lineno(1)))

        elif t[1][1] is 'LONG':
            t[0] = Constant(Constant.LONG, t[1][0], linerange=(t.lineno(1), t.lineno(1)))

        elif t[1][1] is 'LONG_LONG':
            t[0] = Constant(Constant.LONG_LONG, t[1][0], linerange=(t.lineno(1), t.lineno(1)))

    def p_constant_float(self, t):
        """
        constant : FCONST
        """
        self.output_production(t, production_message='constant -> FCONST {}'.format(t[1]))

        t[0] = Constant(Constant.FLOAT, float(t[1]), linerange=(t.lineno(1), t.lineno(1)))

    def p_constant_char(self, t):
        """
        constant : CCONST
        """
        self.output_production(t, production_message='constant -> CCONST ({})'.format(t[1]))

        t[0] = Constant(Constant.CHAR, t[1], linerange=(t.lineno(1), t.lineno(1)))

    #
    # identifier:
    #
    def p_identifier(self, t):
        """
        identifier : ID
        """
        self.output_production(t, production_message='identifier -> ID ({})'.format(t[1]))

        t[0] = t[1]

    #
    # empty:
    #
    def p_empty(self, t):
        """
        empty :
        """
        pass

    #
    # dummy utility productions
    #
    def p_setup_for_program(self, t):
        """
        setup_for_program : empty
        """

        ## Insert library function declarations (prototypes) here ##
        # Super crappy, but it has to be done, blah, blah, blah
        self.compiler_state.symbol_table.insert(library_functions.PrintCharDeclaration)
        self.compiler_state.symbol_table.insert(library_functions.PrintIntDeclaration)
        self.compiler_state.symbol_table.insert(library_functions.PrintStringDeclaration)
        self.compiler_state.symbol_table.insert(library_functions.PrintFloatDeclaration)

    def p_enter_function_scope(self, t):
        """
        enter_function_scope : empty
        """
        self.prod_logger.info('Entering scope {}'.format(len(self.compiler_state.symbol_table.table)))

        self.compiler_state.symbol_table.push()

        for named_parameter in t[-1]['parameters']:
            named_parameter.set_as_parameter()
            self.compiler_state.symbol_table.insert(named_parameter)
            print('OFFSET', named_parameter.activation_frame_offset, named_parameter.size_in_bytes())

    def p_enter_scope(self, t):
        """
        enter_scope : empty
        """
        self.prod_logger.info('Entering scope {}'.format(len(self.compiler_state.symbol_table.table)))

        self.compiler_state.symbol_table.push()

    def p_insert_mode(self, t):
        """
        insert_mode : empty
        """
        self.prod_logger.info('Entering insert mode.')

        self.compiler_state.insert_mode = True

    def p_leave_scope(self, t):
        """
        leave_scope : empty
        """
        self.prod_logger.info('Leaving scope {}'.format(len(self.compiler_state.symbol_table.table) - 1))

        if self.compiler_state.clone_symbol_table_on_next_scope_exit:
            self.prod_logger.info('Cloning the symbol table')
            self.compiler_state.cloned_tables.append(self.compiler_state.symbol_table.clone())
            self.compiler_state.clone_symbol_table_on_next_scope_exit = False

        self.compiler_state.symbol_table.pop()

    def p_lookup_mode(self, t):
        """
        lookup_mode : empty
        """
        self.prod_logger.info('Entering lookup mode.')

        self.compiler_state.insert_mode = False

    # Handles any designated output (other than standard compiler output and warnings).
    #
    # @param self The object pointer
    # @param t The production item with info about the production, including line numbers.
    # @param production_message The production to write. Defaults to 'No Production'.
    #
    # Outputs:
    #   Logger output.
    #
    # Called by the production processing methods.
    #
    def output_production(self, t, production_message='No->Production'):
        message_parts = production_message.split(" -> ")
        production_message = '{rhs:>30} -> {lhs}'.format(rhs=message_parts[0], lhs=message_parts[1])

        line = t.lineno(1)
        if 0 <= line - 1 < len(self.compiler_state.source_code):
            self.prod_logger.source(self.compiler_state.source_lines[line - 1], line=line)
        self.prod_logger.production(production_message)

    # Determines if an object is usable in evaluating a compile-time constant expressions.
    # Called by production handling methods and possibly AST nodes.
    #
    # @param item The item to be checked.
    #
    # Output: Returns True if the item is usable for the constant expression; False otherwise.
    @staticmethod
    def is_a_constant(item):
        valid_types = (Constant, int, float)
        return isinstance(item, valid_types)

    @staticmethod
    def compile_time_evaluable(item):
        return isinstance(item, (Constant, Symbol)) and item.immutable and \
               type_utils.is_integral_type(item.get_resulting_type())

    # Performs compile-time operations to evaluate binary (two-operand) constant expressions.
    # Called by production handling methods.
    #
    # @param left The first operand of the operation.
    # @param operator The operator for the operation.
    # @param right The second operand of the operation.
    #
    # Output: Returns an object representing the (constant) result of the operation.
    @staticmethod
    def perform_binary_operation(left, operator: str, right):

        left_value = left if isinstance(left, int) else left.value
        right_value = right if isinstance(right, int) else right.value

        # right now only returning the value i.e. int.
        # might need to change to return a const ast node instead.
        if operator == '+':
            result = left_value + right_value
        elif operator == '-':
            result = left_value - right_value
        elif operator == '*':
            result = left_value * right_value
        elif operator == '/':
            result = left_value / right_value
        elif operator == '%':
            result = left_value % right_value
        elif operator == '<<':
            result = left_value << right_value
        elif operator == '>>':
            result = left_value >> right_value
        elif operator == '<':
            result = left_value < right_value
        elif operator == '<=':
            result = left_value <= right_value
        elif operator == '>':
            result = left_value > right_value
        elif operator == '>=':
            result = left_value >= right_value
        elif operator == '==':
            result = left_value == right_value
        elif operator == '!=':
            result = left_value != right_value
        elif operator == '&':
            result = left_value & right_value
        elif operator == '|':
            result = left_value | right_value
        elif operator == '^':
            result = left_value ^ right_value
        elif operator == '&&':
            result = 1 if left_value != 0 and right_value != 0 else 0
        elif operator == '||':
            result = 1 if left_value != 0 or right_value != 0 else 0
        else:
            raise Exception('Improper operator provided: ' + operator)

        if isinstance(left, VariableSymbol):
            left_start = left_end = left.lineno
        else:
            left_start = left.linerange[0]
            left_end = left.linerange[1]

        if isinstance(right, VariableSymbol):
            right_start = right_end = right.lineno
        else:
            right_start = right.linerange[0]
            right_end = right.linerange[1]

        first_line = min(left_start, right_start)
        last_line = max(left_end, right_end)

        val_type = Constant.INTEGER if isinstance(result, int) else Constant.FLOAT
        return Constant(val_type, result, linerange=(first_line, last_line))

    # Performs compile-time operations to evaluate unary (one-operand) constant expressions.
    # Called by production handling methods.
    #
    # @param operator The operator for the operation.
    # @param operand The operand of the operation.
    #
    # Output: Returns an object representing the (constant) result of the operation.
    @staticmethod
    def perform_unary_operation(operator: str, operand: Constant):
        value = operand.value if type(operand) is Constant else operand
        if operator == '-':
            return -value
        elif operator == '~':
            return ~value
        elif operator == '!':
            return 1 if value == 0 else 0
        elif operator == '&':
            raise Exception('Used for addressof? How is this handled?')
        elif operator == '*':
            raise Exception('Used for dereferencing? How is this handled?')
        else:
            raise Exception('Improper operator provided: ' + operator)
