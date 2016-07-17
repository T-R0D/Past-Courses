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


def offset_label_immediate(label, offset=0):
    return '{}{}'.format(label, ' + ' + str(offset) if offset != 0 else '')

def offset_from_register_with_immediate(register, offset=0):
    return '{}({})'.format(offset if offset != 0 else '', register)


class BaseMipsInstruction(object):
    NULL = None

    def __init__(self, instruction: str, slot_0: str = NULL, slot_1: str = NULL, slot_2: str = NULL):
        self.instruction = instruction
        self.slot_0 = slot_0
        self.slot_1 = slot_1
        self.slot_2 = slot_2

    def __str__(self):
        return repr(self)

    def __repr__(self):
        """
        __repr__ is supposed to be "unambiguous," but it gets called on all contained objects when you stringify a
        container, so this will be the method that produces the fixed field format string representing the instruction.
        """
        field_format = '{!s:>8}'
        fmt = '{!s:<8}'
        if self.slot_0 is not BaseMipsInstruction.NULL:
            fmt += field_format
        if self.slot_1 is not BaseMipsInstruction.NULL:
            fmt += ', ' + field_format
        if self.slot_2 is not BaseMipsInstruction.NULL:
            fmt += ', ' + field_format

        ret = fmt.format(self.instruction, self.slot_0, self.slot_1, self.slot_2)

        return ret

    def __eq__(self, other):
        return self.instruction == other.instruction and self.slot_0 == other.slot_0 and self.slot_1 == other.slot_1 and \
               self.slot_2 == other.slot_2


#
# ABSTRACT MIPS INSTRUCTION TYPES
#
class Comment(BaseMipsInstruction):
    def __init__(self, instruction, message):
        super(Comment, self).__init__(instruction, message)

    def __str__(self):
        return "## {:^6} ## {}".format(self.instruction, self.slot_0)


class Directive(BaseMipsInstruction):
    def __init__(self, instruction, label=None):
        super(Directive, self).__init__(instruction)
        self.label = label

    def __str__(self):
        return '{}.{}'.format(
            self.label + ': ' if self.label else '',
            type(self).__name__.lower())


#
# INFORMATION
#
class COMMENT(Comment):
    def __init__(self, text):
        super(COMMENT, self).__init__('COMMENT', text)

    def __str__(self):
        return '# {}'.format(self.slot_0)


class SOURCE(Comment):
    def __init__(self, lineno, source_text):
        print(source_text)
        super(SOURCE, self).__init__('SOURCE', '{}: {}'.format(lineno, source_text))


class TAC(Comment):
    def __init__(self, tac_str):
        super(TAC, self).__init__('3AC', tac_str)


#
# MACROS
#
def macro_arg(arg):
    return '%' + str(arg)


class BEGIN_MACRO(object):
    def __init__(self, name, args=None):
        self.name = name
        self.args = args if args else []

    def __str__(self):
        arg_list = ' (' + ', '.join([macro_arg(arg) for arg in self.args]) + ')' if self.args else ''
        return '.macro {}{}'.format(self.name, arg_list)

    def __repr__(self):
        return str(self)

class END_MACRO(object):
    def __init__(self):
        pass

    def __str__(self):
        return '.end_macro\n'

    def __repr__(self):
        return str(self)

class CALL_MACRO(object):
    def __init__(self, macro, call_arguments=None):
        self.macro = macro
        self.call_arguments = call_arguments if call_arguments else []

    def __str__(self):
        arg_list = '(' + ', '.join([str(arg) for arg in self.call_arguments]) + ')'
        return '{}{}'.format(self.macro, arg_list)




#
# PROGRAM SECTION LABELS
#
class DATA(Directive):
    def __init__(self):
        super(DATA, self).__init__('data')

    def __str__(self):
        return '\n.data'


class TEXT(Directive):
    def __init__(self):
        super(TEXT, self).__init__('data')

    def __str__(self):
        return '\n.text'


#
# MEMORY STORAGE DIRECTIVES
#
class SPACE(Directive):
    def __init__(self, label, n_bytes):
        """

        :param label: Use none if this is anonymously reserved space (i.e. no label is given).
        :param n_bytes:
        :return:
        """
        super(SPACE, self).__init__(instruction='spill', label=label)
        self.n_bytes = n_bytes

    def __str__(self):
        return super(SPACE, self).__str__() + ' ' + str(self.n_bytes)


class GLOBAL_WORD(object):
    def __init__(self, label, mem_specifier, value):
        super(GLOBAL_WORD, self).__init__()
        self.label = label
        self.mem_specifier = mem_specifier
        self.value = value

    def __str__(self):
        return '{}: {} {}'.format(self.label, self.mem_specifier, self.value)

    def __repr__(self):
        return str(self)


#
# LABEL
#
class LABEL(BaseMipsInstruction):
    def __init__(self, label):
        super(LABEL, self).__init__('label', label)

    def __str__(self):
        return '\n{}:'.format(self.slot_0)


#
# SYSCALL
#
class SYSCALL(BaseMipsInstruction):
    def __init__(self):
        super(SYSCALL, self).__init__('syscall')


#
# MEMORY
#
li = 'li'
la = 'la'

lb = 'lb'
lhw = 'lhw'
lw = 'lw'

sb = 'sb'
shw = 'shw'
sw = 'sw'


class LI(BaseMipsInstruction):
    def __init__(self, register, immediate):
        super(LI, self).__init__(li, register, immediate)


class LA(BaseMipsInstruction):
    def __init__(self, register, memory_address):
        super(LA, self).__init__(la, register, memory_address)


class LB(BaseMipsInstruction):
    def __init__(self, register, memory_address):
        super(LB, self).__init__(lw, register, memory_address)


class LHW(BaseMipsInstruction):
    def __init__(self, register, memory_address):
        super(LHW, self).__init__(lw, register, memory_address)


class LW(BaseMipsInstruction):
    def __init__(self, register, memory_address):
        super(LW, self).__init__(lw, register, memory_address)

class LWC1(BaseMipsInstruction):
    def __init__(self, register, memory_address):
        super(LWC1, self).__init__('lwc1', register, memory_address)


class SB(BaseMipsInstruction):
    def __init__(self, register, memory_address):
        super(SB, self).__init__(sw, register, memory_address)


class SHW(BaseMipsInstruction):
    def __init__(self, register, memory_address):
        super(SHW, self).__init__(sw, register, memory_address)


class SW(BaseMipsInstruction):
    def __init__(self, register, memory_address):
        super(SW, self).__init__(sw, register, memory_address)


#
# MATH
#
add = 'add'


class ADD(BaseMipsInstruction):
    def __init__(self, sum, addend, augend):
        super(ADD, self).__init__(add, sum, addend, augend)

class ADDIU(BaseMipsInstruction):
    def __init__(self, sum, addend, augend_immediate):
        super(ADDIU, self).__init__('addiu', sum, addend, augend_immediate)

class ADDI(BaseMipsInstruction):
    def __init__(self, sum, addend, augend_immediate):
        super(ADDI, self).__init__('addi', sum, addend, augend_immediate)

class ADDU(BaseMipsInstruction):
    def __init__(self, sum, addend, augend_immediate):
        super(ADDU, self).__init__('addu', sum, addend, augend_immediate)

class SUB(BaseMipsInstruction):
    def __init__(self, difference, minuend, subtrahend):
        super(SUB, self).__init__('sub', difference, minuend, subtrahend)

class SUBIU(BaseMipsInstruction):
    def __init__(self, difference, minuend, subtrahend_immediate):
        super(SUBIU, self).__init__('subiu', difference, minuend, subtrahend_immediate)

class MULU(BaseMipsInstruction):
    def __init__(self, product, multiplicand, multiplier):
        super(MULU, self).__init__('mulu', product, multiplicand, multiplier)

class MUL(BaseMipsInstruction):
    def __init__(self, product, multiplicand, multiplier):
        super(MUL, self).__init__('mul', product, multiplicand, multiplier)

class DIV(BaseMipsInstruction):
    def __init__(self, dividend, divisor):
        super(DIV, self).__init__('div', dividend, divisor)

class SEQ(BaseMipsInstruction):
    def __init__(self, result, src1, src2):
        super(SEQ, self).__init__('seq', result, src1, src2)

class SLT(BaseMipsInstruction):
    def __init__(self, result, src1, src2):
        super(SLT, self).__init__('slt', result, src1, src2)

class SLE(BaseMipsInstruction):
    def __init__(self, result, src1, src2):
        super(SLE, self).__init__('sle', result, src1, src2)

class SGT(BaseMipsInstruction):
    def __init__(self, result, src1, src2):
        super(SGT, self).__init__('sgt', result, src1, src2)

class TLT(BaseMipsInstruction):
    def __init__(self, src1, src2):
        super(TLT, self).__init__('tlt', src1, src2)

class TGE(BaseMipsInstruction):
    def __init__(self, src1, src2):
        super(TGE, self).__init__('tge', src1, src2)

#
# BRANCHING
#
class J(BaseMipsInstruction):
    def __init__(self, target):
        super(J, self).__init__('j', target)

class JAL(BaseMipsInstruction):
    def __init__(self, target):
        super(JAL, self).__init__('jal', target)

class JR(BaseMipsInstruction):
    def __init__(self, register):
        super(JR, self).__init__('jr', register)

class BNE(BaseMipsInstruction):
    def __init__(self, src1, src2, target):
        super(BNE, self).__init__('bne', src1, src2, target)

class BEQZ(BaseMipsInstruction):
    def __init__(self, value, target):
        super(BEQZ, self).__init__('beqz', value, target)


#
# CASTING
#
class CVTSW(BaseMipsInstruction):
    def __init__(self, result, value):
        super(CVTSW, self).__init__('cvt.s.w', result, value)


class CVTWS(BaseMipsInstruction):
    def __init__(self, result, value):
        super(CVTWS, self).__init__('cvt.w.s', result, value)


#
# COPROCESSOR 1
#
class MTC1(BaseMipsInstruction):
    def __init__(self, src, dest):
        super(MTC1, self).__init__('mfc1', src, dest)


class MFC1(BaseMipsInstruction):
    def __init__(self, dest, src):
        super(MFC1, self).__init__('mfc1', dest, src)


if __name__ == '__main__':
    data = DATA()
    print(str(data))
