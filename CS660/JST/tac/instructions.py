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


class BaseInstructionArgument(object):
    def __init__(self):
        pass

    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        return str(self)


class Register(BaseInstructionArgument):
    def __init__(self, register):
        super(Register, self).__init__()
        self.register = register

    def __str__(self):
        return '{}'.format(self.register)


class Immediate(BaseInstructionArgument):
    def __init__(self, value):
        super(Immediate, self).__init__()
        self.value = None

        # TODO: disallow floats? Don't remember how best to handle them with MIPS...

        try:
            self.value = int(value)
        except:
            raise ValueError('Immediate values must be numbers')

    def __str__(self):
        return str(self.value)


class Label(BaseInstructionArgument):
    def __init__(self, label):
        super(Label, self).__init__()
        self.label = label

    def __str__(self):
        return str(self.label)


class Address(BaseInstructionArgument):
    def __init__(self, label=None, int_literal=None, register=None):
        super(Address, self).__init__()

        if label is None and int_literal is None and register is None:
            raise Exception('An Address is undefined without specifying a label, a register, or address literal. ')

        self.label = label
        self.int_literal = int_literal if int_literal and int_literal != 0 else None
        self.register = register

    def __str__(self):
        ret = ''

        if self.label and self.int_literal:
            ret += '{} + {}'.format(self.label, self.int_literal)

        elif self.label:
            ret += str(self.label)

        elif self.int_literal:
            ret += str(self.int_literal)

        if self.register:
            ret += '({})'.format(self.register)

        return ret


# General
DUP = 'DUP'
SOURCE = 'SOURCE'
KICK = 'KICK'

# Program Sections
DATA = 'DATA'
TEXT = 'TEXT'

# Math Operations
# TODO: create versions for both int and floating point?
ADD = 'ADD'
SUB = 'SUB'
MUL = 'MUL'
DIV = 'DIV'
MOD = 'MOD'
NEG = 'NEG'

ADDI = 'ADDI'
SUBI = 'SUBI'

ADDU = 'ADDU'
SUBU = 'SUBU'
MULU = 'MULU'
DIVU = 'DIVU'
MODU = 'MODU'
NEGU = 'NEGU'

ADDIU = 'ADDIU'
SUBIU = 'SUBIU'

ADDS = 'ADDS'
SUBS = 'SUBS'
MULS = 'MULS'
DIVS = 'DIVS'
NEGS = 'NEGS'

# Logical Operations
# These won't correspond directly to MIPS instructions, but it seems like good practice to have them in the 3AC for
# decoupling the instruction from assembly.
LAND = 'LAND'
LOR = 'LOR'

NOT = 'NOT'
EQ = 'EQ'
NE = 'NE'
GT = 'GT'
GE = 'GE'
LT = 'LT'
LE = 'LE'

# Casting
CVTSW = 'CVTSW'
CVTWS = 'CVTWS'

# Assignment
# ASSIGN = 'ASSIGN'

# Memory Accesses
LOAD = 'LOAD'
STORE = 'STORE'
LA = 'LA'
LI = 'LI'

# Control Flow
LABEL = 'LABEL'
JAL = 'JAL'
JR = 'JR'
BR = 'BR'
BREQ = 'BREQ'
BRGT = 'BRGT'
BRLT = 'BRLT'
BRGE = 'BRGE'
BRLE = 'BRLE'
BRNE = 'BRNE'
HALT = 'HALT'

# Procedure/Function Call
ARGS = 'ARGS'
REFOUT = 'REFOUT'
VALOUT = 'VALOUT'
CALL_PROC = 'CALL_PROC'
BEGIN_PROC = 'BEGIN_PROC'
CORP_LLAC = 'END_PROC'

ENTER_PROC = 'PROCENTRY'
EXIT_PROC = 'ENDPROC'
RETURN = 'RETURN'

# Miscellaneous
BOUND = 'BOUND'
ADDR = 'ADDR'
GLOBAL = 'GLOBAL'
STRING = 'STRING'
COMMENT = 'COMMENT'
GLOBLDECL = 'GLOBLDECL'


# Library
PRNTI = 'PRNTI'
PRNTF = 'PRNTF'
PRNTS = 'PRNTS'

READI = 'READI'
READF = 'READF'
READS = 'READS'
