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

import copy
import itertools

import tac.registers as tacr
import tac.instructions as taci
import tac.tac_generation as tac
import ticket_counting.ticket_counters as tickets
from utils import type_utils


class Symbol(object):
    def __init__(self, identifier, lineno, column):
        self.identifier = identifier
        self.lineno = lineno
        self.column = column

        self.global_memory_location = None
        self.activation_frame_offset = None

        # UUID for AST
        self.uuid = tickets.UUID_TICKETS.get()

    def clone(self):
        # TODO Verify this is actually a deepcopy
        return copy.deepcopy(self)


class VariableSymbol(Symbol):
    EMPTY_ARRAY_DIM = None

    def __init__(self, identifier, lineno, column):
        super(VariableSymbol, self).__init__(identifier, lineno, column)

        # In GCC, qualifiers and classes are idempotent
        self.storage_classes = set()
        self.type_qualifiers = set()
        self.type_specifiers = ''

        # Contains size information for array dimensions
        self.array_dims = []

        # Contains qualifiers for pointer dimensions
        self.pointer_dims = []

        # Constant literal value if VariableSymbol is constant
        self.value = None
        self.is_parameter = False

    @property
    def immutable(self):
        return type_utils.CONST in self.type_qualifiers

    @property
    def is_array(self):
        return len(self.array_dims) > 0

    @property
    def array_size(self):
        # the array size in number of elements
        if self.is_array:
            multiplier = 1
            for dim in self.array_dims:
                multiplier *= dim
            return multiplier
        else:
            raise Exception('VariableSymbol is not an array.')

    def add_type_declaration(self, type_declaration):
        self.storage_classes = type_declaration.storage_classes
        self.type_qualifiers = type_declaration.type_qualifiers
        self.type_specifiers = ' '.join(type_declaration.type_specifiers)

    def set_array_dims(self, dims):
        self.array_dims = dims

    def set_pointer_dims(self, dims):
        self.pointer_dims = dims

    def get_resulting_type(self):
        return '{} {}{}'.format(self.type_specifiers,
                                '*' * len(self.pointer_dims),
                                '[]' * len(self.array_dims)).strip()

    def set_as_parameter(self):
        self.is_parameter = True

        if len(self.array_dims) > 0:
            self.array_dims = [self.EMPTY_ARRAY_DIM] * len(self.array_dims)

    # Computes the size in bytes for the VariableSymbol. Pointers will always be 4 bytes (MIPS uses 4 bytes
    # for a word which is always the size of a pointer). VariableSymbols that are arrays will return the
    # size in bytes of one element.
    #
    # @return Size in bytes
    def size_in_bytes(self):
        if len(self.pointer_dims) > 0:
            return 4
        else:
            return int(type_utils.type_size_in_bytes(self.type_specifiers))

    def __str__(self):
        qualifier_str = ' '.join(self.type_qualifiers)
        if len(self.array_dims) > 0:
            array_dim_list = [str(size) if size is not self.EMPTY_ARRAY_DIM
                              else '' for size in self.array_dims]
            array_dim_str = '[' + ']['.join(array_dim_list) + ']'
        else:
            array_dim_str = ''

        pointer_dim_str = '*' * len(self.pointer_dims)

        return '{} {}{}{}{}'.format(qualifier_str,
                                    self.type_specifiers,
                                    (' ' + pointer_dim_str + ' ') if pointer_dim_str != '' else ' ',
                                    self.identifier,
                                    array_dim_str).strip()

    def __repr__(self):
        return str(self)

    # Defined for GraphViz string generation interface compliance
    def name(self):
        return '"{}\\n{}"'.format(str(self), self.uuid)

    # Defined for GraphViz string generation interface compliance
    def to_graph_viz_str(self):
        return '\t{} -> {{}};\n'.format(self.name())

    # Defined for 3AC generation interface compliance
    def to_3ac(self, get_rval=True, include_source=False):
        output = [tac.SOURCE(self.lineno, self.column)]

        # Get ticket to copy memory location
        if type_utils.is_floating_point_type(self.get_resulting_type()):
            value = tickets.FLOAT_REGISTER_TICKETS.get()
        else:
            value = tickets.INT_REGISTER_TICKETS.get()


        if get_rval:
            # load the register with the variable's value

            if self.immutable:
                output.append(tac.LI(value, taci.Immediate(self.value)))


            if self.global_memory_location:
                address = taci.Address(label=self.global_memory_location)

            else:
                address = taci.Address(int_literal=-self.activation_frame_offset, register=tacr.FP)

            output.append(tac.LOAD(value, address, self.size_in_bytes()))

            return {'3ac': output, 'rvalue': value}

        else:
            if self.immutable:
                raise Exception('Immutable symbols should not provide lvalues.')

            # load the register with the address of the variable
            if self.global_memory_location:
                output.append(tac.LA(value, taci.Address(label=self.global_memory_location)))

            else:
                # remember, stack grows downward, so look under the FP

                # LA is probably better for readability than the SUB instruction, but I'm leaving it around in
                # case a bug crops up
                output.append(tac.LA(value, taci.Address(int_literal=-self.activation_frame_offset, register=tacr.FP)))
                # output.append(tac.SUB(value, taci.Register(tacr.FP), self.activation_frame_offset))

                # output.append(tac.ADD(value, taci.Register(tacr.FP), self.activation_frame_offset))
                # pass

            return {'3ac': output, 'lvalue': value}


class FunctionSymbol(Symbol):

    def __init__(self, identifier, lineno, column):
        super(FunctionSymbol, self).__init__(identifier, lineno, column)

        # Initialize as incomplete symbol
        self.finalized = False

        # Defines what parameters this function takes
        # It is a list of VariableSymbols that may or may not have identifiers
        self.named_parameters = []

        self.return_type_pointer_modifiers = []
        self.return_type_specifiers = ''

        # Size in bytes of all variables and params
        # Doesn't count stored registers
        self.activation_frame_size = 0

    def add_return_type_declaration(self, type_declaration):
        self.return_type_specifiers = ' '.join(type_declaration.type_specifiers)

    def set_named_parameters(self, parameter_type_list):
        if self.finalized:
            raise Exception('Attempted redefinition of function {}.'.format(self.identifier))
        self.named_parameters = parameter_type_list

    def arguments_match_parameter_types(self, argument_list):
        if len(self.named_parameters) != len(argument_list):
            return False, 'Argument count does not match required count for function call.'

        if len(argument_list) is 0 and len(self.named_parameters) is 0:
            return True, None

        for parameter, argument in itertools.zip_longest(self.named_parameters, argument_list):

            cast_result, message = type_utils.can_assign(parameter.get_resulting_type(), argument.get_resulting_type())
            if cast_result == type_utils.INCOMPATIBLE_TYPES:
                return False, message + ' for parameter {}'.format(parameter.identifier)

        return True, None

    def get_resulting_type(self):
        return self.return_type_specifiers

    def __str__(self):
        args_as_strings = ''
        if self.named_parameters:
            args_as_strings = ', '.join([str(symbol) for symbol in self.named_parameters])

        return '{} {}({})'.format(self.return_type_specifiers, self.identifier, args_as_strings)

    def __repr__(self):
        return str(self)


class TypeDeclaration(object):
    """
    Not sure if this should remain an AST node, but rather an info collection class that gets disassembled and
    absorbed by the Symbol this contributes to.
    """

    def __init__(self, **kwargs):
        # In GCC, storage classes are idempotent (but there can only be one of each)
        self.storage_classes = set()
        # In GCC, qualifiers are idempotent
        self.type_qualifiers = set()
        # List allows for things like 'long long' and 'long double'
        self.type_specifiers = []
        # Indicated if the type is signed/unsigned
        # None = sign not applicable
        self.type_sign = None

    def add_all_from(self, other_type_declaration):
        # TODO: error checking?
        self.type_specifiers.extend(other_type_declaration.type_specifiers)
        self.type_qualifiers.update(other_type_declaration.type_qualifiers)
        self.storage_classes.update(other_type_declaration.storage_classes)

    def name_arg(self):
        joined = [self.type_sign if self.type_sign else '',
                  ' '.join(self.storage_classes),
                  ' '.join(self.type_qualifiers),
                  ' '.join(self.type_specifiers)]
        return ' '.join([i for i in joined if i is not ''])

    def add_storage_class(self, storage_class_specifier):
        if storage_class_specifier in self.storage_classes:
            raise Exception('Duplication of storage class specifier "{}".'.format(storage_class_specifier))
        self.storage_classes.add(storage_class_specifier)

    def add_type_qualifier(self, type_qualifier):
        self.type_qualifiers.add(type_qualifier)

    def add_type_specifier(self, specifier):
        if specifier == 'unsigned' or specifier == 'signed':
            if self.type_sign is not None:
                raise Exception('Multiple signed/unsigned specifiers not allowed.')
            else:
                self.type_sign = specifier
        else:
            self.type_specifiers.insert(0, specifier)

    def get_type_str(self):
        return (self.type_sign + ' ' if self.type_sign else '') + ' '.join(self.type_specifiers)

    def sizeof(self):
        return type_utils.get_bit_size(self) / 4  # divide by 4 for bytes

    @property
    def immutable(self):
        return 'const' in self.type_qualifiers

    def __str__(self):
        joined = [self.type_sign if self.type_sign else '',
                  ' '.join(self.storage_classes),
                  ' '.join(self.type_qualifiers),
                  ' '.join(self.type_specifiers) if self.type_specifiers else 'Unknown']
        return ' '.join([i for i in joined if i is not ''])

    def __repr__(self):
        return str(self)
