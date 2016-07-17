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
import mips.instructions as mi
import mips.registers as mr


class TestInstructions(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_macros(self):
        self.assertEqual('.macro my_no_arg_macro', str(mi.BEGIN_MACRO('my_no_arg_macro')))
        self.assertEqual('.macro my_argful_macro (%arg_1, %arg_2)',
                         str(mi.BEGIN_MACRO('my_argful_macro', ['arg_1', 'arg_2'])))

        self.assertEqual('.end_macro', str(mi.END_MACRO()))

        self.assertEqual('some_macro()', str(mi.CALL_MACRO('some_macro')))
        self.assertEqual('some_macro_with_args($t0, $t1)', str(mi.CALL_MACRO('some_macro_with_args', [mr.T0, mr.T1])))