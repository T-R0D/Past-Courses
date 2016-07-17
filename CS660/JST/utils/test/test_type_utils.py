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

from utils import type_utils


class TestTypeUtils(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_is_integral_type(self):
        self.assertTrue(type_utils.is_integral_type('char'))
        self.assertTrue(type_utils.is_integral_type('short'))
        self.assertTrue(type_utils.is_integral_type('int'))
        self.assertTrue(type_utils.is_integral_type('long'))
        self.assertTrue(type_utils.is_integral_type('long long'))

        self.assertFalse(type_utils.is_integral_type('void'))
        self.assertFalse(type_utils.is_integral_type('float'))
        self.assertFalse(type_utils.is_integral_type('double'))

        self.assertFalse(type_utils.is_integral_type('long char'))
        self.assertFalse(type_utils.is_integral_type('short long'))

    def test_is_floating_point_type(self):
        self.assertTrue(type_utils.is_floating_point_type('float'))
        self.assertTrue(type_utils.is_floating_point_type('double'))

        self.assertFalse(type_utils.is_floating_point_type('char'))
        self.assertFalse(type_utils.is_floating_point_type('short'))
        self.assertFalse(type_utils.is_floating_point_type('int'))
        self.assertFalse(type_utils.is_floating_point_type('long'))
        self.assertFalse(type_utils.is_floating_point_type('long long'))

        self.assertFalse(type_utils.is_floating_point_type('void'))

        self.assertFalse(type_utils.is_floating_point_type('long char'))
        self.assertFalse(type_utils.is_floating_point_type('short double'))

    def test_get_promoted_type(self):
        self.assertEqual((type_utils.INT, type_utils.CAST_LEFT_UP),
                         type_utils.get_promoted_type(type_utils.CHAR, type_utils.INT))
        self.assertEqual((type_utils.INT, type_utils.CAST_RIGHT_UP),
                         type_utils.get_promoted_type(type_utils.INT, type_utils.CHAR))
        self.assertEqual((type_utils.FLOAT, type_utils.CAST_LEFT_UP),
                         type_utils.get_promoted_type(type_utils.INT, type_utils.FLOAT))
        self.assertEqual((type_utils.UNSIGNED_CHAR, type_utils.CAST_LEFT_UP),
                         type_utils.get_promoted_type(type_utils.CHAR, type_utils.UNSIGNED_CHAR))
        self.assertEqual((type_utils.LONG, type_utils.CAST_UNAFFECTED),
                         type_utils.get_promoted_type(type_utils.LONG, type_utils.LONG))

    def test_type_size_in_bytes(self):
        self.assertEqual(1, type_utils.type_size_in_bytes('char'))
        self.assertEqual(4, type_utils.type_size_in_bytes('int'))
        self.assertEqual(4, type_utils.type_size_in_bytes('float'))
        self.assertEqual(8, type_utils.type_size_in_bytes('double'))
