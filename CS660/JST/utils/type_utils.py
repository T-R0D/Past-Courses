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

#
# TODO: WARNING: NOTHING HAS BEEN DONE HERE TO HANDLE POINTER TYPES, WHICH WE MAY WANT TO DO.
#

# Valid Types
VOID = 'void'
CHAR = 'char'
SIGNED_CHAR = 'signed char'
UNSIGNED_CHAR = 'unsigned char'
SHORT = 'short'
SHORT_INT = 'short int'
SIGNED_SHORT = 'signed short'
SIGNED_SHORT_INT = 'signed short int'
UNSIGNED_SHORT = 'unsigned short'
UNSIGNED_SHORT_INT = 'unsigned short int'
INT = 'int'
SIGNED = 'signed'
SIGNED_INT = 'signed int'
UNSIGNED = 'unsigned'
UNSIGNED_INT = 'unsigned int'
LONG = 'long'
LONG_INT = 'long int'
SIGNED_LONG = 'signed long'
SIGNED_LONG_INT = 'signed long int'
UNSIGNED_LONG = 'unsigned long'
UNSIGNED_LONG_INT = 'unsigned long int'
LONG_LONG = 'long long'
LONG_LONG_INT = 'long long int'
SIGNED_LONG_LONG = 'signed long long'
SIGNED_LONG_LONG_INT = 'signed long long int'
UNSIGNED_LONG_LONG = 'unsigned long long'
UNSIGNED_LONG_LONG_INT = 'unsigned long long int'
FLOAT = 'float'
DOUBLE = 'double'
LONG_DOUBLE = 'long double'

# Type Qualifiers
CONST = 'const'
VOLATILE = 'volatile'


class TypeAttributes(object):
    def __init__(self, rank, bit_size, signed=True, integral=True, floating_point=False):
        self.rank = rank
        self.bit_size = bit_size
        self.signed = signed
        self.integral = integral
        self.floating_point = floating_point

        if floating_point:
            self.signed = False
            self.integral = False


# Note: if two items share the same rank, they are essentially the same type anyway.
# Note: there is a gap in the rankings in case int needs to move down
PRIMITIVE_TYPE_DEFINITIONS = {
    VOID:                    TypeAttributes(rank=0, bit_size=0, signed=False, integral=False, floating_point=False),
    CHAR:                    TypeAttributes(rank=1, bit_size=8),
    SIGNED_CHAR:             TypeAttributes(rank=1, bit_size=8),
    UNSIGNED_CHAR:           TypeAttributes(rank=2, bit_size=8, signed=False),
    SHORT:                   TypeAttributes(rank=3, bit_size=16),
    SHORT_INT:               TypeAttributes(rank=3, bit_size=16),
    SIGNED_SHORT:            TypeAttributes(rank=3, bit_size=16),
    SIGNED_SHORT_INT:        TypeAttributes(rank=3, bit_size=16),
    UNSIGNED_SHORT:          TypeAttributes(rank=4, bit_size=16, signed=False),
    UNSIGNED_SHORT_INT:      TypeAttributes(rank=4, bit_size=16, signed=False),
    INT:                     TypeAttributes(rank=7, bit_size=32),
    SIGNED:                  TypeAttributes(rank=7, bit_size=32),
    SIGNED_INT:              TypeAttributes(rank=7, bit_size=32),
    UNSIGNED:                TypeAttributes(rank=8, bit_size=32, signed=False),
    UNSIGNED_INT:            TypeAttributes(rank=8, bit_size=32, signed=False),
    LONG:                    TypeAttributes(rank=7, bit_size=32),
    LONG_INT:                TypeAttributes(rank=7, bit_size=32),
    SIGNED_LONG:             TypeAttributes(rank=7, bit_size=32),
    SIGNED_LONG_INT:         TypeAttributes(rank=7, bit_size=32),
    UNSIGNED_LONG:           TypeAttributes(rank=8, bit_size=32, signed=False),
    UNSIGNED_LONG_INT:       TypeAttributes(rank=8, bit_size=32, signed=False),
    LONG_LONG:               TypeAttributes(rank=9, bit_size=64),
    LONG_LONG_INT:           TypeAttributes(rank=9, bit_size=64),
    SIGNED_LONG_LONG:        TypeAttributes(rank=9, bit_size=64),
    SIGNED_LONG_LONG_INT:    TypeAttributes(rank=9, bit_size=64),
    UNSIGNED_LONG_LONG:      TypeAttributes(rank=10, bit_size=64, signed=False),
    UNSIGNED_LONG_LONG_INT:  TypeAttributes(rank=10, bit_size=64, signed=False),
    FLOAT:                   TypeAttributes(rank=11, bit_size=32, floating_point=True),
    DOUBLE:                  TypeAttributes(rank=12, bit_size=64, floating_point=True),
    LONG_DOUBLE:             TypeAttributes(rank=13, bit_size=80, floating_point=True)
}


INTEGRAL_TYPES = tuple(filter(
    lambda t: t is not None, [key if value.integral else None for key, value in PRIMITIVE_TYPE_DEFINITIONS.items()]))

FLOATING_POINT_TYPES = tuple(filter(
    lambda t: t is not None, [key if value.floating_point else None for key, value in PRIMITIVE_TYPE_DEFINITIONS.items()]))

CAST_LEFT_UP = 'CAST_LEFT_UP'
CAST_LEFT_DOWN = 'CAST_LEFT_DOWN'
CAST_RIGHT_UP = 'CAST_RESULT_UP'
CAST_RIGHT_DOWN = 'CAST_RESULT_DOWN'
CAST_UNAFFECTED = 'CAST_RESULT_UNAFFECTED'
INCOMPATIBLE_TYPES = 'INCOMPATIBLE_TYPES'


def is_valid_type(declaration):
    type_str = declaration.get_type_str()

    if type_str in PRIMITIVE_TYPE_DEFINITIONS:
        if type_str in INTEGRAL_TYPES:
            if declaration.type_sign is None:
                # Default is a signed value
                declaration.type_sign = 'signed'
        else:
            if declaration.type_sign is not None:
                return False, 'Floating point values cannot be specified as signed/unsigned.'
        return True, None
    else:
        return False, 'Invalid or unknown type ({}).'.format(type_str)


def get_bit_size(type_specifier_str):
    type_attribute = PRIMITIVE_TYPE_DEFINITIONS.get(type_specifier_str, None)

    if type_attribute:
        return type_attribute.bit_size
    else:
        raise Exception('Invalid or unknown type ({}).'.format(type_specifier_str))


def is_primitive_type(type_specifier_str):
    return type_specifier_str in PRIMITIVE_TYPE_DEFINITIONS.keys()


def is_integral_type(type_specifier_str):
    return type_specifier_str in INTEGRAL_TYPES


def is_floating_point_type(type_specifier_str):
    return type_specifier_str in FLOATING_POINT_TYPES


def is_pointer_type(type_specifier_str):
    return type_specifier_str.endswith('*')


def can_assign(left_type, right_type):
    if types_are_compatible(left_type, right_type):
        if left_type == right_type:
            return CAST_UNAFFECTED, None
        elif is_primitive_type(left_type) and is_primitive_type(right_type):
            if PRIMITIVE_TYPE_DEFINITIONS[left_type].rank > PRIMITIVE_TYPE_DEFINITIONS[right_type].rank:
                return CAST_RIGHT_UP, None
            else:
                return CAST_RIGHT_DOWN, None
    else:
        return INCOMPATIBLE_TYPES, "Unable to assign type ({}) to variable of type ({})".format(right_type, left_type)


def get_promoted_type(left_type, right_type):
    if left_type == right_type:
        return left_type, CAST_UNAFFECTED
    elif is_primitive_type(left_type) and is_primitive_type(right_type):
        if PRIMITIVE_TYPE_DEFINITIONS[left_type].rank > PRIMITIVE_TYPE_DEFINITIONS[right_type].rank:
            return left_type, CAST_RIGHT_UP
        else:
            return right_type, CAST_LEFT_UP
    else:
        return None, INCOMPATIBLE_TYPES


def cast_as_required_type(required_type, given_type):
    if required_type == given_type or required_type > given_type:
        return CAST_UNAFFECTED
    elif required_type < given_type:
        # TODO: in 3AC we will have to smash the given value down to fit the required type
        # example, if 256 needs to be stored as a char, return 0
        #             257     ""                          ""   1
        # the process is likely as simple as just keeping the lower bits of the given value (at least for ints, not
        # sure about floats)
        return CAST_RIGHT_DOWN


def types_are_compatible(lhs_type, rhs_type):
    if is_primitive_type(lhs_type) and is_primitive_type(rhs_type):  # TODO: add the logic for pointers and complex types
        return True
    elif lhs_type == rhs_type:
        return True
    else:
        return False


def type_size_in_bits(type_str):
    return PRIMITIVE_TYPE_DEFINITIONS[type_str].bit_size


def type_size_in_bytes(type_str):
    return type_size_in_bits(type_str) / 8
