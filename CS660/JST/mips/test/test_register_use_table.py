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
import mips.registers as mips
import mips.instructions as assembler
from mips.register_management import RegisterUseTable, OutOfSpillMemoryException
from ticket_counting.ticket_counters import INT_REGISTER_TICKETS


SPILL_MEM_LABEL = 'spill'


class TestRegisterUseTable(unittest.TestCase):
    """
    PLEASE NOTE: The tests rely heavily on the setUp() function. In order to make it easy to evaluate the tests, the
                 register_use_table was set up to use a very small set of registers and spill memory. Should these
                 parameters be modified, all of the tests should be revisited to ensure they are performing their
                 tests correctly.
    """

    def setUp(self):
        self.registers_to_use = (mips.T0, mips.T1)
        enough_bytes_to_store_two_registers_and_have_a_swap_space = 12
        self.register_use_table = RegisterUseTable(self.registers_to_use, SPILL_MEM_LABEL,
                                                   enough_bytes_to_store_two_registers_and_have_a_swap_space)

    def tearDown(self):
        pass

    def test_single_register_use(self):

        pseudo_register = INT_REGISTER_TICKETS.get()

        self.assertFalse(pseudo_register in self.register_use_table.lru_cache)

        result = self.register_use_table.acquire(pseudo_register)

        expected_result = {'register': mips.T0, 'code': []}
        self.assertEqual(expected_result, result, 'register $t0 should have been returned and no code required')

        # since no spilling occurred, we should get the same register back with no code needed
        result = self.register_use_table.acquire(pseudo_register)
        self.assertEqual(expected_result, result, 'register $t0 should have been returned and no code required')

        self.register_use_table.release(pseudo_register)
        self.assertFalse(pseudo_register in self.register_use_table.lru_cache)
        self.assertSequenceEqual([mips.T1, mips.T0], self.register_use_table.available_registers)

    def test_register_spill(self):
        pseudo_registers = [INT_REGISTER_TICKETS.get() for _ in range(0, 3)]

        self.register_use_table.acquire(pseudo_registers[0])
        self.register_use_table.acquire(pseudo_registers[1])

        self.assertSequenceEqual([], self.register_use_table.available_registers, 'the registers should be used up')

        expected_result = {'register': mips.T0, 'code': [assembler.SW(mips.T0, assembler.offset_label_immediate(SPILL_MEM_LABEL, 0))]}
        result = self.register_use_table.acquire(pseudo_registers[2])
        self.assertEqual(expected_result, result)

    def test_spill_recovery(self):
        pseudo_registers = [INT_REGISTER_TICKETS.get() for _ in range(0, 3)]

        self.register_use_table.acquire(pseudo_registers[0])
        self.register_use_table.acquire(pseudo_registers[1])

        self.assertSequenceEqual([], self.register_use_table.available_registers, 'the registers should be used up')

        expected_result = {'register': mips.T0, 'code': [assembler.SW(mips.T0, assembler.offset_label_immediate(SPILL_MEM_LABEL, 0))]}
        result = self.register_use_table.acquire(pseudo_registers[2])
        self.assertEqual(expected_result, result)

        # recovering a spilled register's value from memory should not only get the register, but also provide the
        # mips code that swaps the values for the temporaries using the spill memory
        expected_result = {'register': mips.T1, 'code': [assembler.SW(mips.T1, assembler.offset_label_immediate(SPILL_MEM_LABEL, 4)),
                                                         assembler.LW(mips.T1, assembler.offset_label_immediate(SPILL_MEM_LABEL, 0))]}
        result = self.register_use_table.acquire(pseudo_registers[0])
        self.assertEqual(expected_result, result)

    def test_overrun_of_spill_memory_fails(self):
        with self.assertRaises(OutOfSpillMemoryException):
            # 6 is precisely enough to run out of spill memory
            pseudo_registers = [INT_REGISTER_TICKETS.get() for _ in range(0, 6)]
            for pseudo_register in pseudo_registers:
                self.register_use_table.acquire(pseudo_register)

    def test_multiple_spills(self):
        spilled_1 = 'ireg_1'
        spilled_2 = 'ireg_2'
        spiller_1 = 'ireg_3'
        spiller_2 = 'ireg_4'

        self.register_use_table.acquire(spilled_1)
        self.register_use_table.acquire(spilled_2)
        self.register_use_table.acquire(spiller_1)
        self.register_use_table.acquire(spiller_2)

        self.assertTrue(
            spilled_1 in self.register_use_table.spilled_registers.keys() and spilled_2 in
            self.register_use_table.spilled_registers.keys())

    def test_multiple_recoveries(self):
        spilled_1 = 'ireg_1'
        spilled_2 = 'ireg_2'
        spiller_1 = 'ireg_3'
        spiller_2 = 'ireg_4'

        self.register_use_table.acquire(spilled_1)
        self.register_use_table.acquire(spilled_2)
        self.register_use_table.acquire(spiller_1)
        self.register_use_table.acquire(spiller_2)

        self.assertTrue(
            spilled_1 in self.register_use_table.spilled_registers.keys() and spilled_2 in
            self.register_use_table.spilled_registers.keys())

        result = self.register_use_table.acquire(spilled_2)
        expected_result = {'register': mips.T0, 'code': [assembler.SW(mips.T0, assembler.offset_label_immediate(SPILL_MEM_LABEL, 8)),
                                                         assembler.LW(mips.T0, assembler.offset_label_immediate(SPILL_MEM_LABEL, 4))]}
        self.assertEqual(expected_result, result)

        result = self.register_use_table.acquire(spilled_1)
        expected_result = {'register': mips.T1, 'code': [assembler.SW(mips.T1, assembler.offset_label_immediate(SPILL_MEM_LABEL, 4)),
                                                         assembler.LW(mips.T1, assembler.offset_label_immediate(SPILL_MEM_LABEL, 0))]}
        self.assertEqual(expected_result, result)

    def test_release_one_register(self):
        spilled_1 = 'ireg_1'
        spilled_2 = 'ireg_2'
        spiller_1 = 'ireg_3'
        spiller_2 = 'ireg_4'

        self.register_use_table.acquire(spilled_1)
        self.register_use_table.acquire(spilled_2)
        self.register_use_table.acquire(spiller_1)
        self.register_use_table.acquire(spiller_2)

        # test normal release
        self.register_use_table.release(spiller_2)
        self.assertFalse(spiller_2 in self.register_use_table.lru_cache)
        self.assertFalse(spiller_2 in self.register_use_table.spilled_registers)
        self.assertTrue(mips.T1 in self.register_use_table.available_registers)

        # test spilled register release
        self.register_use_table.release(spilled_1)
        self.assertFalse(spiller_2 in self.register_use_table.lru_cache)
        self.assertFalse(spiller_2 in self.register_use_table.spilled_registers)
        self.assertTrue(0 in self.register_use_table.available_spill_memory_words)

    def test_release_all_registers(self):
        spilled_1 = 'ireg_1'
        spilled_2 = 'ireg_2'
        spiller_1 = 'ireg_3'
        spiller_2 = 'ireg_4'

        self.register_use_table.acquire(spilled_1)
        self.register_use_table.acquire(spilled_2)
        self.register_use_table.acquire(spiller_1)
        self.register_use_table.acquire(spiller_2)

        self.register_use_table.release_all()
        self.assertTrue(mips.T0 in self.register_use_table.available_registers)
        self.assertTrue(mips.T1 in self.register_use_table.available_registers)
        self.assertTrue(0 in self.register_use_table.available_spill_memory_words)
        self.assertTrue(4 in self.register_use_table.available_spill_memory_words)
        self.assertTrue(8 in self.register_use_table.available_spill_memory_words)
        self.assertSequenceEqual([], list(self.register_use_table.lru_cache.keys()))
        self.assertSequenceEqual([], list(self.register_use_table.spilled_registers.keys()))

    def test_double_register_use(self):
        self.fail("We aren't supporting two registers for double word purposes yet, cuz it's hard.")
