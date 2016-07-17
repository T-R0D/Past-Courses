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
import re

import pylru
import mips.instructions as mi


class OutOfSpillMemoryException(Exception):
    pass


class InvalidRegisterNameException(Exception):
    pass


class RegisterUseTable(object):
    """ A class for doing the book-keeping of physical register allocation.

    Uses a sequence of registers that are allowed for general use and a labeled range of bytes in order to manage
    register use in a MIPS program. In the event that more registers are needed than are available, registers can be
    'spilled' into the given memory location, for retrieval later if needed. This class maintains a mapping between
    pseudo-registers (aka "temporaries") and real/physical MIPS registers or spill memory.

    Currently, only single register use is supported. That means no 'doubles' or 'long longs'!
    """

    def __init__(self, available_registers, spill_memory_base_label, spill_memory_size, word_size=4):
        """ The constructor.

        Creates a RegisterUseTable with it's allowed set of registers and available spill memory. Arbitrarily sized
        words are allowed, but (sensible) 4 byte words are used by default.

        :param available_registers: A sequence (preferably set or tuple) of the registers the table is allowed to use
                                    without reservation.
        :spill_memory_base_label: The MIPS label where the spill memory is located.
        :param spill_memory_size: The amount of spill memory available in bytes.
        :word_size: The size of a word (register capacity) in bytes.
        :return: A constructed RegisterUseTable object.
        """

        self.spill_mem_base_label = spill_memory_base_label
        self.spill_mem_size = spill_memory_size

        self.available_registers = list(reversed(available_registers))
        self.available_spill_memory_words = list(reversed(range(0, spill_memory_size, word_size)))

        self.lru_cache = pylru.lrucache(size=len(available_registers), callback=self._spill)
        self.spilled_registers = {}

        self._spill_code = []
        self._freed_physical_register = None

    def acquire(self, pseudo_register):
        """ Returns a physical register's name and the assembly required to properly use the register

        Will return a physical register name for use, even if registers must be spilled into memory. Also composes the
        instructions necessary to make use of the register. Everything is returned as a dictionary containing 'register'
        and 'code' keys.

        :param pseudo_register: The name of the pseudo-register.
        :return: A dictionary containing the MIPS register available for use for the pseudo register and a list of
                 MIPS code that was used if spilling occurred.
        """

        if re.match('(i|f)reg_\d+', str(pseudo_register)) is None:
            raise InvalidRegisterNameException('{} is not a valid psuedo-register name'.format(pseudo_register))

        physical_register = None
        code = []

        if pseudo_register in self.lru_cache:
            physical_register = self.lru_cache[pseudo_register]

        elif pseudo_register in self.spilled_registers:
            physical_register, code = self._recover(pseudo_register)

        elif self.available_registers:
            physical_register = self.available_registers.pop()
            self.lru_cache[pseudo_register] = physical_register

        else:  # there are no available registers, so force a spill
            self.lru_cache[pseudo_register] = None
            physical_register = self._freed_physical_register
            self.lru_cache[pseudo_register] = physical_register
            code = self._spill_code

        return {'register': physical_register, 'code': code}

    def release(self, pseudo_register):
        """ Releases any resources held for a pseudo/temporary register (physical register, memory).

        This method should be called whenever possible to free resources for future use.

        :param pseudo_register: The name of the pseudo-register that will no longer be used.
        :return: None.
        """

        if pseudo_register in self.lru_cache:
            self.available_registers.append(self.lru_cache[pseudo_register])
            del self.lru_cache[pseudo_register]

        elif pseudo_register in self.spilled_registers:
            self.available_spill_memory_words.append(self.spilled_registers.pop(pseudo_register))

        else:  # if there is no record of the pseudo-register, there is nothing to do
            pass

    def release_all(self):
        """ Releases all resources (physical registers, memory) for future use.

        Essentially a clear method for the entire table, as opposed to for a single pseudo-register.

        :return: None.
        """
        for pseudo_register in itertools.chain(list(self.lru_cache.keys()), list(self.spilled_registers.keys())):
            self.release(pseudo_register)

    def _spill(self, pseudo_register, physical_register):
        """ A private method for handling the logic for spilling the contents of a physical register into memory.

        Since this method is used as a callback, it cannot return things in the traditional sense. Because of this, the
        "returned" code and physical register are set to two private class variables, so the programmer must take care
        to manage these variables appropriately.

        :param pseudo_register: The name of the pseudo-register to be spilled. Aka 'key' for the purposes of the
                                pylru.lrucache callback.
        :physical_register: The name of the physical register to be freed/reused. Aka 'value' for the purposes of the
                                pylru.lrucache callback.
        :return: None.
        """

        self._spill_code = []
        self._freed_physical_register = physical_register

        if not self.available_spill_memory_words:
            print(self)
            raise OutOfSpillMemoryException()

        spill_offset = self.available_spill_memory_words.pop()

        self.spilled_registers[pseudo_register] = spill_offset

        self._spill_code = [
            mi.COMMENT("spilling psuedo-register {} to free {}".format(pseudo_register, physical_register)),
            mi.SW(physical_register, mi.offset_label_immediate(self.spill_mem_base_label, spill_offset)),
            mi.COMMENT("spill complete")
        ]

    def _recover(self, pseudo_register):
        """ Handles the logic for reversing the spilling of a register.

        :param pseudo_register: The name of the pseudo-register whose spilled value is being recovered.
        :return: A tuple containing the physical register and a list of the MIPS code required to free the physical
                 register for use.
        """

        spill_offset = self.spilled_registers.pop(pseudo_register)
        physical_register = None
        code = [
            mi.COMMENT("recovering pseudo-register {}".format(pseudo_register))
        ]

        if self.available_registers:
            physical_register = self.available_registers.pop()
            self.lru_cache[pseudo_register] = physical_register

        else:
            self.lru_cache[pseudo_register] = None
            physical_register = self._freed_physical_register

            code.extend(self._spill_code)

            self.lru_cache[pseudo_register] = physical_register

            code.append(
                mi.LW(physical_register, mi.offset_label_immediate(self.spill_mem_base_label, spill_offset)))

        self.available_spill_memory_words.append(spill_offset)

        return physical_register, code

    def __str__(self):
        """ A method for convenient representation of the table.

        Concatenates the informative members of the class together to summarize the state of the RegisterUseTable. The
        string does neglect some of the finer details, but should be sufficient for most necessary insights to the
        table.

        :return: A string representing the state of the RegisterUseTable.
        """
        lru_cache_str = '{' + ', '.join([str(item[0]) + ': ' + str(item[1]) for item in self.lru_cache.items()]) + '}'

        ret = 'lru cache:             {}\n' \
              'spilled registers:     {}\n' \
              'available registers:   {}\n' \
              'available spill words: {}'.format(lru_cache_str, self.spilled_registers, self.available_registers,
                                                 len(self.available_spill_memory_words))


        return ret

    def __repr__(self):
        """ See __str__.

        :return: The result of the __str__ method.
        """
        return str(self)
