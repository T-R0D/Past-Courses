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
import tac.tac_generation as tac


class TestTacGeneration(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_add(self):
        instruction = tac.ADD('temp_0000', 'temp_0001', 'temp_0002')

        self.assertEqual('ADD, temp_0000, temp_0001, temp_0002', str(instruction))
        self.assertEqual('ADD            , temp_0000      , temp_0001      , temp_0002      ', repr(instruction))

    def test_comment(self):
        instruction = tac.COMMENT("Hi! I'm a comment")

        self.assertEqual("COMMENT: Hi! I'm a comment", str(instruction))
        self.assertEqual("# Hi! I'm a comment", repr(instruction))

    def test_label(self):
        instruction = tac.LABEL('ENTER_LOOP')

        self.assertEqual("LABEL, ENTER_LOOP, None, None", str(instruction))
        self.assertEqual("ENTER_LOOP:", repr(instruction))
