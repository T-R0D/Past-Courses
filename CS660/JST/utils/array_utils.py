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
import itertools


EMPTY_ARRAY_DIM = None


def array_dimensions_are_valid(array_dimensions):
    for dimension in array_dimensions[:-1]:
        if dimension == EMPTY_ARRAY_DIM:
            raise Exception('Invalid array dimension. Only the last dimension may be undefined.')
        elif not isinstance(dimension, int):
            raise Exception(
                'Only integral types may be used to specify array dimensions. {} is unacceptable'.format(dimension))

    last_dimension = array_dimensions[-1]
    if not last_dimension == EMPTY_ARRAY_DIM and not isinstance(last_dimension, int):
        raise Exception(
            'Only integral types may be used to specify array dimensions. {} is unacceptable'.format(last_dimension))

    return True


# TODO: general algorithm of an array dereference
# width * j + i
