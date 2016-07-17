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

import itertools
from ast.base_ast_node import BaseAstNode
from symbol_table.symbol import VariableSymbol
from utils import type_utils
from utils import operator_utils
import ticket_counting.ticket_counters as tickets
from tac.tac_generation import *
import tac.instructions as taci
import tac.registers as tacr


WORD_SIZE = 4


##
# Node for referencing an array through subscripts.
##
class ArrayReference(BaseAstNode):
    """
    Requires: Information about the array being dereferenced and the results of any expressions indicating the
              indices where dereferencing is occuring, stored in temporary registers. An indication of if an rvalue
              or lvalue should be produced should be given (since an lvalue likely means a pointer should be returned
              vs. a raw value).
    Output:   In the case of an lvalue, a temporary register with the address of the memory of the element of interest,
              in the case of an rvalue, a temporary register containing the actual value of the element.
    """

    def __init__(self, symbol, subscripts=None, **kwargs):
        super(ArrayReference, self).__init__(**kwargs)

        self.symbol = symbol
        self.subscripts = subscripts if subscripts else []

    def get_resulting_type(self):
        """
        For interface compliance with the other expression nodes.
        """
        type_str = self.symbol.get_resulting_type()
        # TODO figure out the resulting type after subscripting - Shubham (sg-variable-symbol)
        first_open_bracket = type_str.index('[')
        return type_str[:first_open_bracket - 1]

    def size_in_bytes(self):
        if len(self.symbol.array_dims) != len(self.subscripts):
            raise Exception("Attempt to get size in bytes of a not-fully dereferenced array.")
        else:
            return self.symbol.size_in_bytes()

    @property
    def immutable(self):
        return self.symbol.immutable

    def check_subscripts(self):
        if len(self.symbol.array_dims) != len(self.subscripts):
            return False, 'Symbol has {} dimensions, but only {} were provided.'.format(len(self.symbol.array_dims),
                                                                                        len(self.subscripts))
        return True, None

    def name(self, arg=None):
        return super(ArrayReference, self).name(arg=self.symbol.identifier)

    @property
    def children(self):
        children = []
        children.extend(self.subscripts)
        return tuple(children)

    def to_3ac(self, get_rval=True, include_source=False):
        _3ac = [SOURCE(self.linerange[0], self.linerange[1])]
        dim_count = len(self.symbol.array_dims)

        if dim_count is 0:
            raise Exception('There was an error. Subscripting a non-array symbol.')

        # Initialize offset_reg to the value of the first subscript
        subscript_tac = self.subscripts[0].to_3ac(get_rval=True)
        if '3ac' in subscript_tac:
            _3ac.extend(subscript_tac['3ac'])

        # The first subscript is the initial value for the subscript
        offset_reg = subscript_tac['rvalue']

        # range(a,b) is [inclusive, exclusive)
        for i in range(0, dim_count - 1):
            if self.symbol.is_parameter:

                # Allocate a new ticket to get the address offset from FP
                dimension_reg = tickets.INT_REGISTER_TICKETS.get()
                _3ac.append(LOAD(dimension_reg,
                                 taci.Address(int_literal=-(self.symbol.activation_frame_offset + 8 + (4 * (i + 1))),
                                              register=tacr.FP),
                                 WORD_SIZE))
                _3ac.append(MUL(offset_reg, offset_reg, dimension_reg))
                _3ac.append(KICK(dimension_reg))

            else:
                _3ac.append(MUL(offset_reg, offset_reg, self.symbol.array_dims[i + 1]))

            # Add the 3AC to load the subscript
            subscript_tac = self.subscripts[i + 1].to_3ac(get_rval=True)
            if '3ac' in subscript_tac:
                _3ac.extend(subscript_tac['3ac'])

            # Add the subscript to the offset
            _3ac.append(ADDU(offset_reg, offset_reg, subscript_tac['rvalue']))

            # Kick the temporary
            _3ac.append(KICK(subscript_tac['rvalue']))

        # Offset by symbol size
        _3ac.append(MUL(offset_reg, offset_reg, self.symbol.size_in_bytes()))

        # Allocate two new registers
        base_address_reg = tickets.INT_REGISTER_TICKETS.get()
        end_address_reg = tickets.INT_REGISTER_TICKETS.get()

        # Check bounds
        if self.symbol.is_parameter:

            _3ac.append(LOAD(
                    base_address_reg,
                    taci.Address(int_literal=-(self.symbol.activation_frame_offset), register=tacr.FP),
                    WORD_SIZE))
            _3ac.append(LOAD(
                    end_address_reg,
                    taci.Address(int_literal=-(self.symbol.activation_frame_offset + WORD_SIZE), register=tacr.FP),
                    WORD_SIZE))
            _3ac.append(SUB(end_address_reg, base_address_reg, end_address_reg))
            _3ac.append(SUB(offset_reg, base_address_reg, offset_reg))

            # Check base_address_reg >= offset_reg > end_address_reg (because upside-down stack)
            _3ac.append(BOUND(offset_reg, base_address_reg, end_address_reg))

        else:

            # Compute the size in bytes for the entire array
            array_size_in_bytes = self.symbol.array_size * self.symbol.size_in_bytes()

            if self.symbol.global_memory_location:

                _3ac.append(LI(base_address_reg, self.symbol.global_memory_location))
                _3ac.append(LI(end_address_reg, self.symbol.global_memory_location - array_size_in_bytes))
                # TODO This doesn't look right. Global is a 154235345 type number right?
                _3ac.append(SUB(offset_reg, self.symbol.global_memory_location, offset_reg))

            else:
                _3ac.append(LA(
                        base_address_reg,
                        taci.Address(int_literal=-(self.symbol.activation_frame_offset), register=tacr.FP)))
                _3ac.append(ADDI(end_address_reg, base_address_reg, -array_size_in_bytes))
                _3ac.append(SUB(offset_reg, base_address_reg, offset_reg))

            # Check base_address_reg >= offset_reg > end_address_reg (because upside-down stack)
            _3ac.append(BOUND(offset_reg, base_address_reg, end_address_reg))

        # Kick the temporaries
        _3ac.append(KICK(base_address_reg))
        _3ac.append(KICK(end_address_reg))

        # Return the appropriate dict
        if get_rval:
            _3ac.append(LOAD(offset_reg, taci.Address(register=offset_reg), self.symbol.size_in_bytes()))
            return {'3ac': _3ac, 'rvalue': offset_reg}
        else:
            return {'3ac': _3ac, 'lvalue': offset_reg}


class Assignment(BaseAstNode):
    """
    Requires: An lvalue register produced by the expression of the thing being assigned to and an rvalue register
              containing the value being assigned.
    Output:   A temporary rvalue register that contains the value that was assigned. Perhaps a slight optimization would
              be to just return that rvalue register that the RHS gave this node to start with?
    """

    def __init__(self, operator, lvalue, rvalue, **kwargs):
        super(Assignment, self).__init__(**kwargs)

        self.operator = operator
        self.lvalue = lvalue
        self.rvalue = rvalue

    def get_resulting_type(self):
        return self.lvalue.get_resulting_type()

    def name(self, arg=None):
        return super(Assignment, self).name(arg=operator_utils.operator_to_name(self.operator))

    @property
    def children(self):
        children = [self.lvalue, self.rvalue]
        return tuple(children)

    def to_3ac(self, get_rval=True, include_source=False):
        _3ac = [SOURCE(self.linerange[0], self.linerange[1])]

        # Get memory address of lvalue by calling to3ac on lvalue
        left = self.lvalue.to_3ac(get_rval=False)
        lval = left['lvalue']
        if '3ac' in left:
            _3ac.extend(left['3ac'])

        # Get rvalue by calling to3ac on rvalue
        right = self.rvalue.to_3ac(get_rval=True)
        rval = right['rvalue']
        if '3ac' in right:
            _3ac.extend(right['3ac'])

        # Store the rvalue at the lvalue address
        _3ac.append(STORE(
                taci.Register(rval),
                taci.Address(register=lval),
                self.lvalue.size_in_bytes()))
        _3ac.append(KICK(lval))

        return {'3ac': _3ac, 'rvalue': rval}


class BinaryOperator(BaseAstNode):
    """
    Requires: Two rvalue registers that contain the values to be operated on.
    Output:   An rvalue register containing the value of the result of the operation.
    """

    def __init__(self, operator, lvalue, rvalue, **kwargs):
        super(BinaryOperator, self).__init__(**kwargs)

        self.operator = operator
        self.lvalue = lvalue
        self.rvalue = rvalue

    def get_resulting_type(self):
        lvalue_type = self.lvalue.get_resulting_type()
        rvalue_type = self.rvalue.get_resulting_type()

        # TODO (Shubham) We may just need to make a table according to operations
        # For example:
        #   Comparison operators always result in an integral type even though operands can be non-integrals
        #   Shift operators require an int on the left (right can be downcast to an int), so they result in an 'int'
        #   +/-/*/div operations result in a highest precision type
        #   Mod operation always has to return an integral type
        #   Bitwise AND, OR, and XOR require integers and have to return an integral type
        resulting_type, cast_result = type_utils.get_promoted_type(lvalue_type, rvalue_type)

        return resulting_type

    def name(self, arg=None):
        return super(BinaryOperator, self).name(arg=operator_utils.operator_to_name(self.operator))

    @property
    def children(self):
        children = [self.lvalue, self.rvalue]
        return tuple(children)

    def to_3ac(self, get_rval=True, include_source=False):
        output = [SOURCE(self.linerange[0], self.linerange[1])]

        # get memory address of lvalue by calling to3ac on lvalue
        left = self.lvalue.to_3ac(get_rval=True)
        lval = left['rvalue']
        output.extend(left['3ac'])

        # get memory address of rvalue by calling to3ac on rvalue
        right = self.rvalue.to_3ac(get_rval=True)
        rval = right['rvalue']
        output.extend(right['3ac'])

        # get temporary register
        # TODO: Add in checking for int or float so can pull correct ticket
        reg = tickets.INT_REGISTER_TICKETS.get()

        # TODO: NEED TO ADD IN OPTIONS BASED ON TYPE OF TICKET PULLED HERE
        # determine operator type and call correct 3ac instruction with registers

        if self.operator == '+':
            output.append(ADD(reg, lval, rval))

        elif self.operator == '-':
            output.append(SUB(reg, lval, rval))

        elif self.operator == '*':
            output.append(MUL(reg, lval, rval))

        elif self.operator == '/':
            output.append(DIV(reg, lval, rval))

        elif self.operator == '%':
            output.append(MOD(reg, lval, rval))

        elif self.operator == '>>':
            # output.append(   (reg, lval, rval))
            # TODO: what function is this?
            raise NotImplementedError('Please implement the {} binary operator to3ac method.'.format(self.operator))

        elif self.operator == '<<':
            # output.append(   (reg, lval, rval))
            # TODO: what function is this?
            raise NotImplementedError('Please implement the {} binary operator to3ac method.'.format(self.operator))

        elif self.operator == '<':
            output.append(LT(reg, lval, rval))

        elif self.operator == '<=':
            output.append(LE(reg, lval, rval))

        elif self.operator == '>':
            output.append(GT(reg, lval, rval))

        elif self.operator == '>=':
            output.append(GE(reg, lval, rval))

        elif self.operator == '==':
            output.append(EQ(reg, lval, rval))

        elif self.operator == '!=':
            output.append(NE(reg, lval, rval))

        elif self.operator == '&':
            # output.append(  (reg, lval, rval))
            # TODO: what function is this?
            raise NotImplementedError('Please implement the {} binary operator to3ac method.'.format(self.operator))

        elif self.operator == '|':
            # output.append(   (reg, lval, rval))
            # TODO: what function is this?
            raise NotImplementedError('Please implement the {} binary operator to3ac method.'.format(self.operator))

        elif self.operator == '^':
            # output.append(    (reg, lval, rval))
            # TODO: what function is this?
            raise NotImplementedError('Please implement the {} binary operator to3ac method.'.format(self.operator))

        elif self.operator == '&&':
            output.append(LAND(reg, lval, rval))

        elif self.operator == '||':
            output.append(LOR(reg, lval, rval))

        # TODO: since don't have the value since not calculating anything, can't store it to the table yet
        # register_allocation_table[reg] = value

        # Kick the temporaries
        output.append(KICK(lval))
        output.append(KICK(rval))
        return {'3ac': output, 'rvalue': reg}


class Cast(BaseAstNode):
    """
    Requires: An rvalue of the value to be casted.
    Output:   An rvalue register containing the casted value (which might have changed over the course of the casting
              process).
    """

    def __init__(self, to_type, expression, **kwargs):
        super(Cast, self).__init__(**kwargs)

        self.to_type = to_type
        self.expression = expression

    @property
    def immutable(self):
        return False

    def get_resulting_type(self):
        return self.to_type

    def name(self, arg=None):
        return super(Cast, self).name(self.to_type)

    @property
    def children(self):
        children = [self.expression]
        return tuple(children)

    def to_3ac(self, get_rval=True, include_source=False):
        # since returns cast value, should only return rvalue
        _3ac = [SOURCE(self.linerange[0], self.linerange[1])]

        expression_type = self.expression.get_resulting_type()
        expression_result = self.expression.to_3ac()
        _3ac.extend(expression_result['3ac'])

        return_register = expression_result['rvalue']

        # if same, don't cast
        if self.to_type != expression_type:
            if type_utils.is_integral_type(self.to_type):
                new_register = tickets.INT_REGISTER_TICKETS.get()
                _3ac.append(CVTSW(new_register, return_register))
                return_register = new_register
            else:
                new_register = tickets.FLOAT_REGISTER_TICKETS.get()
                _3ac.append(CVTWS(new_register, return_register))
                return_register = new_register

        return {'3ac': _3ac, 'rvalue': return_register}


class CompoundStatement(BaseAstNode):
    """
    Requires: None.
    Output:   None.

    It is unlikely that this node will produce any 3AC, but will simply amalgamate the code generated by its children.
    """

    def __init__(self, declaration_list=None, statement_list=None, **kwargs):
        super(CompoundStatement, self).__init__(**kwargs)

        self.declaration_list = declaration_list
        self.statement_list = statement_list

    @property
    def children(self):
        children = []
        if self.declaration_list is not None:
            children.extend(self.declaration_list)
        if self.statement_list is not None:
            children.extend(self.statement_list)
        return tuple(children)

    def to_3ac(self, include_source=False):
        output = [SOURCE(self.linerange[0], self.linerange[1])]

        # Gen 3ac for declaration_list
        if self.declaration_list is not None:
            for item in self.declaration_list:
                result = item.to_3ac()
                output.extend(result['3ac'])

                # Since the result is unused after this point, kick it out
                if 'rvalue' in result:
                    output.append(KICK(result['rvalue']))
                if 'lvalue' in result:
                    output.append(KICK(result['lvalue']))

        # Gen 3ac for statement_list
        if self.statement_list is not None:
            for item in self.statement_list:

                result = item.to_3ac()
                output.extend(result['3ac'])

                # Since the result is unused after this point, kick it out
                if 'rvalue' in result:
                    output.append(KICK(result['rvalue']))
                if 'lvalue' in result:
                    output.append(KICK(result['lvalue']))

        return {'3ac': output}


class Declaration(BaseAstNode):
    """
    Requires: The symbol information of the declaration.
    Output:   Probably no direct output in the form of temporary registers, but the memory assigned for the
              thing should be recorded somewhere.
    """

    def __init__(self, symbol, initializer=None, **kwargs):
        super(Declaration, self).__init__(**kwargs)

        self.symbol = symbol
        self.initializer = initializer

    def name(self, arg=None):
        arg = self.symbol.get_resulting_type() + ' ' + self.symbol.identifier
        return super(Declaration, self).name(arg)

    @property
    def children(self):
        children = []
        if self.initializer is not None:
            children.append(self.initializer)
        return tuple(children)

    def to_3ac(self, include_source=False):
        _3ac = [SOURCE(self.linerange[0], self.linerange[1])]

        if self.initializer is None:
            return {'3ac': _3ac}

        if self.symbol.global_memory_location:
            if self.initializer:
                if not self.initializer.immutable:
                    raise Exception('Initializers to global objects must be constants.')

                _3ac.append(GLOBLDECL(self.symbol.global_memory_location, WORD_SPEC, self.initializer.value))

            else:
                _3ac.append(GLOBLDECL(self.symbol.global_memory_location, WORD_SPEC))

            return {'3ac': _3ac}

        # Get a register that points to the variable's memory so we can initialize it
        lvalue = tickets.INT_REGISTER_TICKETS.get()
        if self.symbol.global_memory_location:
            base_address = self.symbol.global_memory_location
            _3ac.append(LI(lvalue, base_address))
        else:
            # Remember, stacks grow down, so go below FP
            _3ac.append(LA(lvalue, taci.Address(int_literal=-self.symbol.activation_frame_offset, register=tacr.FP)))

        # If the initializer is a list (for arrays)
        if isinstance(self.initializer, list):

            # Loop through initializer and store
            self.initializer = self.initializer[:min(len(self.initializer), self.symbol.array_dims[0])]
            for item in self.initializer:

                # Load the value
                item_tac = item.to_3ac(get_rval=True)
                if '3ac' in item_tac:
                    _3ac.extend(item_tac['3ac'])

                # Store the value into memory, kick the register, and move to next
                _3ac.append(STORE(item_tac['rvalue'], taci.Address(register=lvalue), self.symbol.size_in_bytes()))

                # Kick the temporaries
                if 'rvalue' in item_tac:
                    _3ac.append(KICK(item_tac['rvalue']))
                if 'lvalue' in item_tac:
                    _3ac.append(KICK(item_tac['lvalue']))

                # Go to the next index / offset by subtracting one element size
                _3ac.append(ADDI(lvalue, lvalue, -self.symbol.size_in_bytes()))

        else:

            # Load the value
            item_tac = self.initializer.to_3ac(get_rval=True)
            if '3ac' in item_tac:
                _3ac.extend(item_tac['3ac'])

            # Store the value into memory, kick the register,
            _3ac.append(STORE(item_tac['rvalue'], taci.Address(register=lvalue), self.symbol.size_in_bytes()))

            # Kick the temporaries
            if 'rvalue' in item_tac:
                _3ac.append(KICK(item_tac['rvalue']))
            if 'lvalue' in item_tac:
                _3ac.append(KICK(item_tac['lvalue']))

        # Kick the base address
        _3ac.append(KICK(lvalue))
        return {'3ac': _3ac}


class FileAST(BaseAstNode):
    """ Root node of the AST.
    Requires: None.
    Output:   None.

    Simply amalgamates 3AC. Might not produce any code other than standard boilerplate.
    """

    def __init__(self, external_declarations, compiler_state, **kwargs):
        super(FileAST, self).__init__(**kwargs)

        self.external_declarations = external_declarations if external_declarations else []
        self.compiler_state = compiler_state if compiler_state else None

    @property
    def children(self):
        children = []
        children.extend(self.external_declarations)
        return tuple(children)

    def to_3ac(self, include_source=False):
        _3ac = []
        global_data_declarations = []
        function_definitions = []

        for external_declaration in self.external_declarations:
            if isinstance(external_declaration, FunctionDefinition):
                function_definitions.append(external_declaration)
            else:
                global_data_declarations.append(external_declaration)

        # Start the data section
        _3ac.append(DATA())
        for external_declaration in global_data_declarations:
            result = external_declaration.to_3ac()
            _3ac.extend(result['3ac'])

        # Start the text section
        _3ac.append(TEXT())

        # Insert the call to main
        _3ac.append(JAL(self.compiler_state.main_function.identifier))

        # End the call to main
        _3ac.append(BR('PROG_END'))

        for function_definition in function_definitions:
            result = function_definition.to_3ac()
            _3ac.extend(result['3ac'])

        # Add a program end label
        _3ac.append(LABEL('PROG_END'))

        # Convert 3AC to a string
        _3ac_as_str = ''
        last_line = 0
        for item in _3ac:
            if item.instruction == 'SOURCE':
                if include_source:
                    if item.dest > last_line:
                        for lineno in range(last_line, item.dest):
                            _3ac_as_str += '# ' + self.compiler_state.source_lines[lineno] + '\n'
                        last_line = item.dest
            else:
                _3ac_as_str += str(item) + '\n'

        if include_source:
            for lineno in range(last_line, len(self.compiler_state.source_lines)):
                print('# ' + self.compiler_state.source_lines[lineno] + '\n')

        return _3ac, _3ac_as_str

    def to_graph_viz_str(self):
        return 'digraph {\n' + super(FileAST, self).to_graph_viz_str() + '}'


class FunctionCall(BaseAstNode):
    """
    Requires: An rvalue for each parameter that this node will then take appropriate actions to cast and copy into the
              activation frame.
    Output:   An rvalue in a temporary register. Take care to copy the value from the value that is stored in the MIPS
              designated return value register.
    """

    def __init__(self, function_symbol, arguments=None, **kwargs):
        super(FunctionCall, self).__init__(**kwargs)

        self.function_symbol = function_symbol
        self.arguments = arguments if arguments else []

    def get_resulting_type(self):
        return self.function_symbol.get_resulting_type()

    def name(self, arg=None):
        return super(FunctionCall, self).name(arg=self.function_symbol.identifier)

    @property
    def children(self):
        children = []
        children.extend(self.arguments)
        return tuple(children)

    def to_3ac(self, get_rval=True, include_source=False):
        _3ac = []
        return_type = self.function_symbol.get_resulting_type()

        if type_utils.is_floating_point_type(return_type):
            rvalue = tickets.FLOAT_REGISTER_TICKETS.get()
        else:
            rvalue = tickets.INT_REGISTER_TICKETS.get()

        # Call the prologue macro
        _3ac.append(CALL_PROC(self.function_symbol.identifier, self.function_symbol.activation_frame_size))

        # Copy the argument values into
        for parameter_template, argument in itertools.zip_longest(self.function_symbol.named_parameters,
                                                                  self.arguments):
            # evaluate the argument expression
            # get register with value of arg
            arg_result = argument.to_3ac(get_rval=True)
            _3ac.extend(arg_result['3ac'])
            arg_rvalue = arg_result['rvalue']

            arg_type = type_utils.INT if arg_rvalue[0] == 'i' else type_utils.FLOAT
            param_type = type_utils.INT if type_utils.is_integral_type(return_type) else type_utils.FLOAT

            # Get casted value if necessary
            if arg_type != param_type:
                if type_utils.is_integral_type(param_type):
                    new_register = tickets.INT_REGISTER_TICKETS.get()
                    _3ac.append(CVTSW(new_register, arg_rvalue))
                    _3ac.append(KICK(arg_rvalue))
                    arg_rvalue = new_register
                else:
                    new_register = tickets.FLOAT_REGISTER_TICKETS.get()
                    _3ac.append(CVTWS(new_register, arg_rvalue))
                    _3ac.append(KICK(arg_rvalue))
                    arg_rvalue = new_register

            # store value at memory location indicated by parameter_template
            # offset = parameter_template.activation_frame_offset
            #  ^
            # likely unnecessary since we are moving the stack pointer as we push arguments
            # if a bug crops up, be sure to check this out
            offset = 0

            if isinstance(argument, VariableSymbol) and argument.is_array:
                # Kick the old register in case it was a float register
                _3ac.append(KICK(arg_rvalue))

                # Get a new integer register
                arg_rvalue = tickets.INT_REGISTER_TICKETS.get()
                _3ac.append(LA(
                        arg_rvalue,
                        taci.Address(int_literal=-parameter_template.activation_frame_offset, register=tacr.FP)))

                # Store the base address
                _3ac.append(STORE(arg_rvalue, taci.Address(int_literal=offset, register=tacr.SP), WORD_SIZE))
                _3ac.append(SUB(taci.Register(tacr.SP), taci.Register(tacr.SP), WORD_SIZE))

                # Store the total array size
                _3ac.append(LI(arg_rvalue, argument.size_in_bytes() * argument.array_size))
                _3ac.append(STORE(arg_rvalue, taci.Address(int_literal=offset, register=tacr.SP), WORD_SIZE))
                _3ac.append(SUB(taci.Register(tacr.SP), taci.Register(tacr.SP), WORD_SIZE))

                # Store the size of each dimension
                for dim in argument.array_dims:
                    _3ac.append(LI(arg_rvalue, dim))
                    _3ac.append(STORE(arg_rvalue, taci.Address(int_literal=offset, register=tacr.SP), WORD_SIZE))
                    _3ac.append(SUB(taci.Register(tacr.SP), taci.Register(tacr.SP), WORD_SIZE))

            else:
                # Store the value and move the stack pointer
                _3ac.append(STORE(arg_rvalue, taci.Address(int_literal=offset, register=tacr.SP), WORD_SIZE))
                _3ac.append(SUB(taci.Register(tacr.SP), taci.Register(tacr.SP), WORD_SIZE))

            # Kick out the temporary at the end of the argument iterating loop
            _3ac.append(KICK(arg_rvalue))

        # Jump to function body
        _3ac.append(JAL(self.function_symbol.identifier))

        # The function will jump back to this address at this point

        # Call the epilogue macro
        _3ac.append(CORP_LLAC(self.function_symbol.activation_frame_size))

        # Copy the return value before it gets obliterated
        _3ac.append(ADD(rvalue, taci.Register(tacr.RV), taci.Register(tacr.ZERO)))

        # TODO: handle double word returns if we get there
        return {'3ac': _3ac, 'rvalue': rvalue}


class FunctionDeclaration(BaseAstNode):
    """
    Requires: The symbol information of the declaration.
    Output:   Nothing direct
    """

    def __init__(self, function_symbol, arguments=None, **kwargs):
        super(FunctionDeclaration, self).__init__(**kwargs)

        self.function_symbol = function_symbol

        self.identifier = function_symbol.identifier
        self.arguments = arguments if arguments else []

    def name(self, arg=None):
        arg = self.function_symbol.get_resulting_type() + ' ' + self.identifier
        return super(FunctionDeclaration, self).name(arg)

    @property
    def children(self):
        children = []
        children.extend(self.arguments)
        return tuple(children)

    def to_3ac(self, include_source=False):
        return {'3ac': []}


class FunctionDefinition(BaseAstNode):
    """
    Requires: Information about its params and declarations so that it can build its activation frame appropriately
    Output:   Nothing other than the 3AC
    """

    def __init__(self, function_symbol, identifier, arguments, body, **kwargs):
        super(FunctionDefinition, self).__init__(**kwargs)

        self.function_symbol = function_symbol
        self.identifier = identifier
        self.body = body
        self.arguments = arguments if arguments else []

    def name(self, arg=None):
        arg = self.function_symbol.get_resulting_type() + ' ' + self.identifier
        return super(FunctionDefinition, self).name(arg)

    @property
    def children(self):
        children = []
        children.extend(self.arguments)
        if self.body:
            children.append(self.body)
        return tuple(children)

    def to_3ac(self, include_source=False):
        _tac = [SOURCE(self.linerange[0], self.linerange[1]), LABEL(self.function_symbol.identifier)]

        parameter_size = 0
        for symbol in self.function_symbol.named_parameters:
            if symbol.is_array:
                parameter_size += WORD_SIZE * (1 + 1 + len(symbol.array_dims))
            else:
                parameter_size += symbol.size_in_bytes()
                # parameter_size += EXPECTED_WORD_SIZE

        # be wary of this, it seems too easy to be correct
        _tac.append(ENTER_PROC(local_variable_size=self.function_symbol.activation_frame_size - parameter_size))

        # Generate 3AC for body (always a compound statement)
        if self.body:
            result = self.body.to_3ac()
            _tac.extend(result['3ac'])

        # Jump back the caller
        _tac.append(EXIT_PROC())

        return {'3ac': _tac}


class If(BaseAstNode):
    """
    Requires: 3AC from child nodes.
    Output:   No direct output, just the 3AC associated with the conditional checking and branching.
    """

    def __init__(self, conditional, if_true, if_false, **kwargs):
        super(If, self).__init__(**kwargs)

        self.conditional = conditional
        self.if_true = if_true
        self.if_false = if_false

    @property
    def children(self):
        children = [self.conditional]
        if self.if_true:
            children.append(self.if_true)
        if self.if_false:
            children.append(self.if_false)
        return tuple(children)

    def to_3ac(self, include_source=False):
        output = [SOURCE(self.linerange[0], self.linerange[1])]

        # Get two labels
        label_true = tickets.IF_TRUE_TICKETS.get()
        label_false = tickets.IF_FALSE_TICKETS.get()
        label_end = tickets.ENDIF_TICKETS.get()

        # Gen 3ac for conditional
        result = self.conditional.to_3ac()
        output.extend(result['3ac'])

        # Branch based on the contents of the result register of the conditional
        reg = result['rvalue']
        output.append(BRNE(label_true, reg, taci.Register(tacr.ZERO)))
        output.append(LABEL(label_false))

        # Gen 3AC for false branch
        if self.if_false:
            result = self.if_false.to_3ac()
            output.extend(result['3ac'])

        # Add jump to end of conditional
        output.append(BR(label_end))

        # Gen 3AC for true branch
        output.append(LABEL(label_true))
        if self.if_true:
            result = self.if_true.to_3ac()
            output.extend(result['3ac'])

        # Dump end label
        output.append(LABEL(label_end))
        output.append(KICK(reg))

        return {'3ac': output}


class InitializerList(BaseAstNode):
    def __init__(self, initializers=None, **kwargs):
        super(InitializerList, self).__init__(**kwargs)

        self.initializers = initializers if initializers else []

    @property
    def children(self):
        children = []
        children.extend(self.initializers)
        return tuple(children)

    def to_3ac(self, include_source=False):
        raise NotImplementedError('Please implement the {}.to_3ac(self) method.'.format(type(self).__name__))


class IterationNode(BaseAstNode):
    """ Node for all forms of structured iteration (for, while, and do...while).
    Requires: 3AC from child nodes. In the parser, the members of this node should have been initialized in such a way
              as to be correct, i.e. an error was thrown if the continuation condition was not given, so we are OK to
              make assumptions now, i.e. a missing continuation condition indicates an infinite for loop.
    Output:   No direct output, just the 3AC associated with the conditional checking and branching.
    """

    def __init__(self, is_pre_test_loop, initialization_expression, stop_condition_expression, increment_expression,
                 body_statements=None, **kwargs):
        super(IterationNode, self).__init__(**kwargs)

        self.is_pre_test_loop = is_pre_test_loop

        self.initialization_expression = initialization_expression
        self.stop_condition_expression = stop_condition_expression
        self.increment_expression = increment_expression
        self.body_statements = body_statements

    @property
    def children(self):
        children = []
        if self.initialization_expression:
            children.append(self.initialization_expression)
        if self.stop_condition_expression:
            children.append(self.stop_condition_expression)
        if self.increment_expression:
            children.append(self.increment_expression)
        if self.body_statements:
            children.append(self.body_statements)
        return tuple(children)

    def to_3ac(self, include_source=False):
        output = [SOURCE(self.linerange[0], self.linerange[1])]
        condition_check_label = tickets.LOOP_CONDITION_TICKETS.get()
        condition_ok_label = tickets.LOOP_BODY_TICKETS.get()
        loop_exit_label = tickets.LOOP_EXIT_TICKETS.get()

        # Check for pre-test loop
        if not self.is_pre_test_loop:
            output.append(BR(condition_ok_label))

        # Initialize
        if self.initialization_expression:
            result = self.initialization_expression.to_3ac()
            output.extend(result['3ac'])

            # Since the result is unused after this point, kick it out
            if 'rvalue' in result:
                output.append(KICK(result['rvalue']))
            if 'lvalue' in result:
                output.append(KICK(result['lvalue']))

        # Add condition check label
        output.append(LABEL(condition_check_label))

        # Check condition
        if self.stop_condition_expression:
            condition_tac = self.stop_condition_expression.to_3ac()

            # If condition is false
            output.extend(condition_tac['3ac'])
            output.append(BRNE(condition_ok_label, condition_tac['rvalue'], taci.Register(tacr.ZERO)))
            output.append(BR(loop_exit_label))

        # Add condition okay label
        output.append(LABEL(condition_ok_label))

        # Add loop instructions
        if self.body_statements:
            result = self.body_statements.to_3ac()
            output.extend(result['3ac'])

        # Add increment expressions
        if self.increment_expression:
            result = self.increment_expression.to_3ac()
            output.extend(result['3ac'])

            # Since the result is unused after this point, kick it out
            if 'rvalue' in result:
                output.append(KICK(result['rvalue']))
            if 'lvalue' in result:
                output.append(KICK(result['lvalue']))

        # Add loop instruction
        output.append(BR(condition_check_label))

        # Add loop exit label
        output.append(LABEL(loop_exit_label))

        return {'3ac': output}


class Label(BaseAstNode):
    """
    Requires: Only its own name.
    Output:   The 3AC for a label.
    """

    def __init__(self, label_name, body_statement, **kwargs):
        super(Label, self).__init__(**kwargs)

        self.label_name = label_name
        self.body_statement = body_statement

    def name(self, arg=None):
        return super(Label, self).name(arg=self.label_name)

    @property
    def children(self):
        children = [self.body_statement]
        return tuple(children)

    def to_3ac(self, include_source=False):
        raise NotImplementedError('Please implement the {}.to_3ac(self) method.'.format(type(self).__name__))


class Return(BaseAstNode):
    """
    Requires: An appropriately initialized register containing information on where to jump to at the return of the
              function; an rvalue register containing the result of the associated expression. If the expression is
              empty, then the register should contain the value 0 (zero).
    Output:   None, per se, but it needs to store that rvalue in the designated MIPS return register.
    """

    def __init__(self, expression, **kwargs):
        super(Return, self).__init__(**kwargs)

        self.expression = expression

    def get_resulting_type(self):
        """
        For interface compliance with the other expression nodes.
        """
        return self.expression.get_resulting_type()

    @property
    def children(self):
        children = []
        if self.expression:
            children.append(self.expression)
        return tuple(children)

    def to_3ac(self, include_source=False):
        output = [SOURCE(self.linerange[0], self.linerange[1])]

        prev_result = {}
        if self.expression:
            prev_result = self.expression.to_3ac()
            output.extend(prev_result['3ac'])

        output.append(RETURN(prev_result.get('rvalue', tacr.ZERO)))
        return {'3ac': output}


class UnaryOperator(BaseAstNode):
    """
    Requires: An lvalue for the operation to operate on.
    Output:   An rvalue that is either the result of the operation or the value of the symbol before the operation,
              depending on if the operator is a pre- or post- one.
    """

    def __init__(self, operator, pre, expression, **kwargs):
        super(UnaryOperator, self).__init__(**kwargs)

        self.operator = operator
        self.expression = expression
        self.pre = pre

    def get_resulting_type(self):
        return self.expression.get_resulting_type()

    def name(self, arg=None):
        return super(UnaryOperator, self).name(arg=operator_utils.operator_to_name(self.operator))

    @property
    def children(self):
        children = [self.expression]
        return tuple(children)

    def to_3ac(self, get_rval=False, include_source=False):
        output = [SOURCE(self.linerange[0], self.linerange[1])]

        # Get memory location of expression by calling to3ac function
        result = (self.expression.to_3ac(get_rval=False))
        if '3ac' in result:
            output.extend(result['3ac'])
        lvalue = result['lvalue']

        # Get the rvalue
        rvalue = tickets.INT_REGISTER_TICKETS.get()
        output.append(LOAD(rvalue, taci.Address(register=lvalue), WORD_SIZE))

        # if this is a post-increment, copy the register with the current value of the register so we can return that
        # before the plusplus happens
        if not self.pre:

            # Copy the contents of the rvalue before the operator is applied
            # It will be returned while the actual variable is updated
            rvalue_copy = tickets.INT_REGISTER_TICKETS.get()
            output.append(ADD(rvalue_copy, rvalue, taci.Register(tacr.ZERO)))


            # Determine correct operator and apply to register
            if self.operator == '++':
                output.append(ADDIU(rvalue, rvalue, 1))
            if self.operator == '--':
                output.append(SUBI(rvalue, rvalue, 1))

            # Store updated value and kick the lvalue & rvalue
            output.append(STORE(rvalue, taci.Address(register=lvalue), WORD_SIZE))
            output.append(KICK(lvalue))
            output.append(KICK(rvalue))

            return {'3ac': output, 'rvalue': rvalue_copy}

        else:

            # Determine correct operator and apply to register
            if self.operator == '++':
                output.append(ADDIU(rvalue, rvalue, 1))
            if self.operator == '--':
                output.append(SUBI(rvalue, rvalue, 1))

            # Store updated value and kick the lvalue
            output.append(STORE(rvalue, taci.Address(register=lvalue), WORD_SIZE))
            output.append(KICK(lvalue))

            return {'3ac': output, 'rvalue': rvalue}


class Constant(BaseAstNode):
    """
    Requires: The value of the constant/constant expression as received from the parser.
    Output:   An rvalue register containing the value of the constant.
    """
    CHAR = 'char'
    INTEGER = 'int'
    LONG = 'long'
    LONG_LONG = 'long long'
    FLOAT = 'float'
    DOUBLE = 'double'

    def __init__(self, type_, value, **kwargs):
        super(Constant, self).__init__(**kwargs)

        self.type_declaration = type_
        self.value = value

    def get_resulting_type(self):
        """
        For interface compliance with the other expression types.
        """
        return self.type_declaration

    def name(self, **kwargs):
        return super(Constant, self).name(arg=str(self.value))

    @property
    def immutable(self):
        return True

    @staticmethod
    def is_integral_type(source):
        return type_utils.is_integral_type(source.get_resulting_type())

    @property
    def children(self):
        children = []
        return tuple(children)

    def to_3ac(self, get_rval=True, include_source=False):
        output = [SOURCE(self.linerange[0], self.linerange[1])]

        # TODO: handle floats
        reg = tickets.INT_REGISTER_TICKETS.get()
        output.append(LI(reg, self.value))

        # Since value is constant, an rvalue is always returned
        return {'3ac': output, 'rvalue': reg}
