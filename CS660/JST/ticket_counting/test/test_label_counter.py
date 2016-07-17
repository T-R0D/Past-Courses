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
from ticket_counting.ticket_counters import UUID_TICKETS, LABEL_TICKETS, INT_REGISTER_TICKETS, FLOAT_REGISTER_TICKETS


class TestLabelCounter(unittest.TestCase):
    def setUp(self):
        UUID_TICKETS.next_value = 0
        LABEL_TICKETS.next_value = 0
        INT_REGISTER_TICKETS.next_value = 0
        FLOAT_REGISTER_TICKETS.next_value = 0

    def tearDown(self):
        UUID_TICKETS.next_value = 0
        LABEL_TICKETS.next_value = 0
        INT_REGISTER_TICKETS.next_value = 0
        FLOAT_REGISTER_TICKETS.next_value = 0

    def test_get(self):
        self.assertEqual("00000", UUID_TICKETS.get())
        self.assertEqual("label_00000", LABEL_TICKETS.get())
        self.assertEqual("ireg_00000", INT_REGISTER_TICKETS.get())
        self.assertEqual("freg_00000", FLOAT_REGISTER_TICKETS.get())

        self.assertEqual("00001", UUID_TICKETS.get())
        self.assertEqual("label_00001", LABEL_TICKETS.get())
        self.assertEqual("ireg_00001", INT_REGISTER_TICKETS.get())
        self.assertEqual("freg_00001", FLOAT_REGISTER_TICKETS.get())
