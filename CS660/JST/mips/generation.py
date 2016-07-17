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
import sys
import mips.register_management as mrm
import mips.registers as mr
import mips.instructions as mi
from tac.tac_generation import TacInstruction
import tac.tac_generation as tac_gen
import tac.instructions as taci
import tac.registers as tacr
import mips.macros as mm
import mips.configurations as config
import mips.library_functions as library_functions


WORD_SIZE = 4


class MipsGenerator(object):
    """ An object to drive translation from 3AC to MIPS.

    This object conducts the translation from 3AC to MIPS and contains the results within itself.
    """

    def __init__(self, compiler_state, temporary_registers=config.TEMPROARY_REGISTER_SET,
                 spill_mem_label=config.SPILL_MEM_LABEL, spill_mem_size=config.SPILL_MEM_SIZE, inject_source=False,
                 inject_3ac=False, inject_comments=False):
        """ The constructor.

        Constructs the MipsGenerator object.

        :param compiler_state: Contains any parameters required for the execution of the compiler program.
        :param temporary_registers: A set of the registers that are allowed to be used as temporaries.
        :param spill_mem_label: The label to use in the generated MIPS to allocate register spilling memory.
        :param spill_mem_size: The amount of memory to make available for register spilling.
        :return: A constructed MipsGenerator.
        """

        self.compiler_state = compiler_state
        self.next_source_line_to_inject = 0
        self.inject_source = inject_source
        self.inject_3ac = inject_3ac
        self.inject_comments = inject_comments

        self.source_tac = []
        self.mips_output = []
        self.register_table = mrm.RegisterUseTable(temporary_registers, spill_mem_label, spill_mem_size)

    def load(self, source_tac):
        """ Supply the 3AC the generator should use to generate MIPS assembly code from.

        :param source_tac: An iterable of 3AC items to translate from.
        :return: None.
        """
        self.source_tac = source_tac

    def dumps(self):
        """ Dump the generated MIPS code to a string.

        Iterates over all generated MIPS code and produces a single string with one MIPS instruction (or
        source/comment/original 3AC item) per line. The lines are separated by new-lines.

        :return: The constructed string with one MIPS instruction per line.
        """
        return '\n'.join([str(mips_instruction) for mips_instruction in self.mips_output])

    def dump(self, file_name=None):
        """ Dumps the generated MIPS to a file.

        Uses dumps() internally.

        :param file_name: The name of the file to dump the generated code to. Defaults to the stdout.
        :return: None.
        """
        file = None
        if file_name:
            file = open(file_name)
        else:
            file = sys.stdout

        file.write(self.dumps())

        if file_name:
            file.close()

    def translate_tac_to_mips(self):
        """ Iterates over the set source 3AC and generates new MIPS code.

        This function iterates over the given 3AC and produces MIPS that is stored internally. This method is
        essentially a poor man's parser that calls methods for each type of 3AC as appropriate.

        :return: None.
        """

        self.mips_output.extend(mm.SAVE_REGISTER_MACRO.definition())
        self.mips_output.extend(mm.RESTORE_REGISTER_MACRO.definition())

        self.mips_output.extend(mm.SAVE_SPILL_MEM_MACRO.definition())
        self.mips_output.extend(mm.RESTORE_SPILL_MEM_MACRO.definition())

        self.mips_output.extend(mm.CALLEE_FUNCTION_PROLOGUE_MACRO.definition())
        self.mips_output.extend(mm.CALLEE_FUNCTION_EPILOGUE_MACRO.definition())

        self.mips_output.extend(mm.CALLER_FUNCTION_PROLOGUE_MACRO.definition())
        self.mips_output.extend(mm.CALLER_FUNCTION_EPILOGUE_MACRO.definition())

        self.mips_output.extend(mm.LAND_MACRO.definition())
        self.mips_output.extend(mm.LOR_MACRO.definition())

        for instruction in self.source_tac:

            assert (isinstance(instruction, TacInstruction))
            if instruction.instruction == taci.SOURCE:
                self._source(instruction.dest)

            elif instruction.instruction == taci.COMMENT:
                if self.inject_comments:
                    self._comment(instruction)

            else:
                if self.inject_3ac:
                    self.mips_output.append(mi.TAC(str(instruction)))

                if instruction.instruction == taci.DATA:
                    self._data_section(instruction)

                elif instruction.instruction == taci.TEXT:
                    self._text_section(instruction)

                elif instruction.instruction == taci.LABEL:
                    self._label(instruction)

                elif instruction.instruction == taci.KICK:
                    self._kick(instruction)

                elif instruction.instruction == taci.CALL_PROC:
                    self._call_procedure(instruction)


                elif instruction.instruction == taci.ENTER_PROC:
                    self._enter_procedure(instruction)

                elif instruction.instruction == taci.EXIT_PROC:
                    self._exit_procedure(instruction)

                elif instruction.instruction == taci.CORP_LLAC:
                    self._take_control_back_from_procedure_call(instruction)

                elif instruction.instruction == taci.LI:
                    self._load_immediate(instruction)

                elif instruction.instruction == taci.LOAD:
                    self._load(instruction)

                elif instruction.instruction == taci.LA:
                    self._load_address(instruction)

                elif instruction.instruction == taci.STORE:
                    self._store(instruction)

                elif instruction.instruction == taci.BR:
                    self._branch(instruction)

                elif instruction.instruction == taci.BRNE:
                    self._branch_not_equal(instruction)

                # elif instruction.instruction == taci.ASSIGN:
                #     self._assign(instruction)

                elif instruction.instruction == taci.JR:
                    self._jump_to_register_value(instruction)

                elif instruction.instruction == taci.JAL:
                    self._jump_and_link(instruction)

                elif instruction.instruction == taci.ADDIU:
                    self._add_immediate_unsigned(instruction)

                elif instruction.instruction == taci.ADDI:
                    self._add_immediate(instruction)

                elif instruction.instruction == taci.ADDU:
                    self._add_unsigned(instruction)

                elif instruction.instruction == taci.ADD:
                    self._add(instruction)

                elif instruction.instruction == taci.SUB:
                    self._sub(instruction)

                elif instruction.instruction == taci.SUBI:
                    self._sub_immediate(instruction)

                elif instruction.instruction == taci.MULU:
                    self._multiply_unsigned(instruction)

                elif instruction.instruction == taci.MUL:
                    self._multiply(instruction)

                elif instruction.instruction == taci.MOD:
                    self._modulo(instruction)

                elif instruction.instruction == taci.DIV:
                    self._div(instruction)

                # elif instruction.instruction == taci.MFHI:
                #     self._move_from_high(instruction)

                elif instruction.instruction == taci.LAND:
                    self._logical_and(instruction)

                elif instruction.instruction == taci.LOR:
                    self._logical_or(instruction)


                elif instruction.instruction == taci.EQ:
                    self._equality(instruction)

                elif instruction.instruction == taci.LT:
                    self._less_than(instruction)

                elif instruction.instruction == taci.LE:
                    self._less_than_equal(instruction)

                elif instruction.instruction == taci.GT:
                    self._greater_than(instruction)

                elif instruction.instruction == taci.BOUND:
                    self._bound(instruction)

                elif instruction.instruction == taci.RETURN:
                    self._return(instruction)


                # elif instruction.instruction == taci.LT:


                elif instruction.instruction == taci.CVTSW:
                    self._convert_float_to_int(instruction)

                elif instruction.instruction == taci.CVTWS:
                    self._convert_int_to_float(instruction)

                elif instruction.instruction == taci.GLOBLDECL:
                    self._global_variable_declaration(instruction)

                else:
                    raise NotImplementedError("{} 3AC -> MIPS is not implemented!".format(instruction.instruction))

        # slip in anything else that was missed/not included by 3AC
        end_of_program_label = self.mips_output.pop()

        self._source(len(self.compiler_state.source_lines))

        #~ Add any pre-defined library functions here ~#
        # This is a really crappy way of doing things, but it will have to do since our compiler only handles
        # 'single file' programs. The declarations (prototypes) must be added to the symbol table in the Parser, in a
        # dummy production "setup_for_program" to allow calls to these functions.

        self.mips_output.extend(library_functions.PrintCharDefinition.get_mips())
        self.mips_output.extend(library_functions.PrintIntDefinition.get_mips())
        self.mips_output.extend(library_functions.PrintStringDefinition.get_mips())
        self.mips_output.extend(library_functions.PrintFloatDefinition.get_mips())

        # add the end of the program, including returning the result of main
        self.mips_output.append(end_of_program_label)
        self.mips_output.append(mi.ADD(mr.A0, mr.V0, mr.ZERO))
        self.mips_output.append(mi.LI(mr.V0, 17))
        self.mips_output.append(mi.SYSCALL())

    def _source(self, last_line):
        if self.inject_source:

            if last_line > self.next_source_line_to_inject:
                for line in range(self.next_source_line_to_inject, last_line):
                    if line < len(self.compiler_state.source_lines):
                        source_line = self.compiler_state.source_lines[line]
                        self.mips_output.append(mi.SOURCE(line, source_line))

                self.next_source_line_to_inject = last_line

    def _comment(self, t):
        self.mips_output.append(mi.COMMENT(text=t.dest))

    def _data_section(self, t):
        self.mips_output.append(mi.DATA())

        # this may need to be moved elsewhere, but it's here for now to make it work
        self.mips_output.append(mi.SPACE(label=config.SPILL_MEM_LABEL, n_bytes=config.SPILL_MEM_SIZE))

    def _text_section(self, t):
        self.mips_output.append(mi.TEXT())

        # this may need to be moved elsewhere, but it's here for now to make it work
        self.mips_output.append(mi.ADD(mr.FP, mr.SP, mr.ZERO))
        self.mips_output.append(mi.ADD(mr.A0, mr.FP, mr.ZERO))

    def _label(self, t):
        self.mips_output.append(mi.LABEL(t.dest))

    def _kick(self, instruction):
        self.register_table.release(instruction.dest)

    def _call_procedure(self, t):
        # push the temporary registers on the stack
        self.mips_output.append(mm.CALLER_FUNCTION_PROLOGUE_MACRO.call())

        # the function call AST node will handle pushing arguments onto the stack and jumping and linking to the
        # function

    def _enter_procedure(self, t):
        self.register_table.release_all()
        self.mips_output.append(mm.CALLEE_FUNCTION_PROLOGUE_MACRO.call(int(t.dest / mr.WORD_SIZE)))

    def _exit_procedure(self, t):
        self.mips_output.append(mm.CALLEE_FUNCTION_EPILOGUE_MACRO.call())

    def _take_control_back_from_procedure_call(self, t):
        self.mips_output.append(mm.CALLER_FUNCTION_EPILOGUE_MACRO.call())

    def _return(self, t):
        result = self.register_table.acquire(t.dest)
        self.mips_output.extend(result['code'])
        register = result['register']
        self.mips_output.append(mi.ADD(mr.V0, register, mr.ZERO))

        # TODO: we are considering have returns jump to the code that actually exits a function, this is good for now
        self.mips_output.append(mm.CALLEE_FUNCTION_EPILOGUE_MACRO.call())

    def _load_immediate(self, t):
        result = self.register_table.acquire(t.dest)


        self.mips_output.extend(result['code'])
        self.mips_output.append(mi.LI(result['register'], t.src1))

    def _load(self, t):
        destination = self.get_resulting_argument(t.dest)
        address = self.get_resulting_argument(t.src1)

        if t.src2 is 1:
            self.mips_output.append(mi.LB(destination, address))
        elif t.src2 is 2:
            self.mips_output.append(mi.LHW(destination, address))
        elif t.src2 is 4:
            self.mips_output.append(mi.LW(destination, address))

    def _store(self, t):
        # TODO: ensure that the address is handled for all of the variants
        # ($t2), 100($t2), 100, label, label + immediate, label($t2), label + immediate($t2)

        if not isinstance(t.dest, (taci.Register, str)):  # TODO: refactor so the str is unnecessary
            raise ValueError('The first argument of a store instruction must be a register.')

        if not isinstance(t.src1, taci.Address):
            raise ValueError('The second argument of a store instruction must be an address')

        content = self.get_resulting_argument(t.dest)
        address = self.get_resulting_argument(t.src1)

        # Content and then address
        if t.src2 is 1:
            self.mips_output.append(mi.SB(content, address))
        elif t.src2 is 2:
            self.mips_output.append(mi.SHW(content, address))
        elif t.src2 is 4:
            self.mips_output.append(mi.SW(content, address))

    def _load_address(self, t):
        destination = self.get_resulting_argument(t.dest)
        address = self.get_resulting_argument(t.src1)

        self.mips_output.append(mi.LA(destination, address))

    #
    # def _assign(self, t):
    #     dest_register = None
    #     src1_register = None
    #
    #     result = self.register_table.acquire(t.src1)
    #     self.mips_output.extend(result['code'])
    #     dest_register = result['register']
    #
    #     if t.src1 not in tac_gen.CONSTANT_REGISTERS:
    #         result = self.register_table.acquire(t.src2)
    #         self.mips_output.extend(result['code'])
    #         src1_register = result['register']
    #     else:
    #         src1_register = self.tac_special_register_to_mips(t.src2)
    #
    #     self.mips_output.append(mi.SW(dest_register, src1_register))


    def _add_immediate_unsigned(self, t):
        sum = self.get_resulting_argument(t.dest)
        addend = self.get_resulting_argument(t.src1)
        augend = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.ADDIU(sum, addend, augend))

    def _add_immediate(self, t):
        sum = self.get_resulting_argument(t.dest)
        addend = self.get_resulting_argument(t.src1)
        augend = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.ADDI(sum, addend, augend))

    def _add_unsigned(self, t):
        sum = self.get_resulting_argument(t.dest)
        addend = self.get_resulting_argument(t.src1)
        augend = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.ADDU(sum, addend, augend))

    def _add(self, t):
        sum = self.get_resulting_argument(t.dest)
        addend = self.get_resulting_argument(t.src1)
        augend = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.ADD(sum, addend, augend))

    def _sub(self, t):
        dest_register = self.get_resulting_argument(t.dest)
        src1_register = self.get_resulting_argument(t.src1)
        src2_register = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.SUB(dest_register, src1_register, src2_register))

    def _sub_immediate(self, t):
        dest_register = self.get_resulting_argument(t.dest)
        src1_register = self.get_resulting_argument(t.src1)
        src2_register = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.SUB(dest_register, src1_register, src2_register))


    def _multiply_unsigned(self, t):
        product = self.get_resulting_argument(t.dest)
        multiplicand = self.get_resulting_argument(t.src1)
        multiplier = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.MULU(product, multiplicand, multiplier))

    def _multiply(self, t):
        product = self.get_resulting_argument(t.dest)
        multiplicand = self.get_resulting_argument(t.src1)
        multiplier = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.MUL(product, multiplicand, multiplier))

    def _modulo(self, t):
        result = self.get_resulting_argument(t.dest)
        multiplicand = self.get_resulting_argument(t.src1)
        multiplier = self.get_resulting_argument(t.src2)

        self.mips_output.append("DIV          " + multiplicand + ',' + multiplier)
        self.mips_output.append("MFHI         " + result)

    def _div(self, t):
        result = self.get_resulting_argument(t.dest)
        dividend = self.get_resulting_argument(t.src1)
        divisor = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.DIV(dividend, divisor))
        self.mips_output.append("MFLO         " + result)

    def _logical_and(self, t):
        result = self.get_resulting_argument(t.dest)
        dividend = self.get_resulting_argument(t.src1)
        divisor = self.get_resulting_argument(t.src2)

        self.mips_output.append(mm.LAND_MACRO.call(dividend, divisor))
        self.mips_output.append(mi.ADD(result, mr.A2, mr.ZERO))

    def _logical_or(self, t):
        result = self.get_resulting_argument(t.dest)
        dividend = self.get_resulting_argument(t.src1)
        divisor = self.get_resulting_argument(t.src2)

        self.mips_output.append(mm.LOR_MACRO.call(dividend, divisor))
        self.mips_output.append(mi.ADD(result, mr.A2, mr.ZERO))


    def _equality(self, t):
        dest_register = self.get_resulting_argument(t.dest)
        src1_register = self.get_resulting_argument(t.src1)
        src2_register = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.SEQ(dest_register, src1_register, src2_register))

    def _less_than(self, t):
        dest_register = self.get_resulting_argument(t.dest)
        src1_register = self.get_resulting_argument(t.src1)
        src2_register = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.SLT(dest_register, src1_register, src2_register))

    def _less_than_equal(self, t):
        dest_register = self.get_resulting_argument(t.dest)
        src1_register = self.get_resulting_argument(t.src1)
        src2_register = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.SLE(dest_register, src1_register, src2_register))

    def _greater_than(self, t):
        dest_register = self.get_resulting_argument(t.dest)
        src1_register = self.get_resulting_argument(t.src1)
        src2_register = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.SGT(dest_register, src1_register, src2_register))

    def _bound(self, t):
        dest_register = self.get_resulting_argument(t.dest)
        src1_register = self.get_resulting_argument(t.src1)
        src2_register = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.TLT(src1_register, dest_register))
        self.mips_output.append(mi.TGE(src2_register, dest_register))

    def _branch(self, t):
        self.mips_output.append(mi.J(t.dest))

    def _branch_not_equal(self, t):
        target = t.dest
        src1_register = self.get_resulting_argument(t.src1)
        src2_register = self.get_resulting_argument(t.src2)

        self.mips_output.append(mi.BNE(src1_register, src2_register, target))

    def _jump_to_register_value(self, t):
        register = self.get_resulting_argument(t.dest)
        self.mips_output.append(mi.JR(register))

    def _jump_and_link(self, t):
        self.mips_output.append(mi.JAL(t.dest))

    def _convert_int_to_float(self, t):
        register = t.dest
        coproc1_register = t.src1
        self.mips_output.append(mi.MTC1(register, coproc1_register))
        self.mips_output.append(mi.CVTSW(coproc1_register, coproc1_register))

    def _convert_float_to_int(self, t):
        coproc1_register = t.dest
        register = t.src1
        self.mips_output.append(mi.CVTWS(coproc1_register, coproc1_register))
        self.mips_output.append(mi.MFC1(register, coproc1_register))



    def _global_variable_declaration(self, t):
        self.mips_output.append(mi.GLOBAL_WORD(t.dest, t.src1, t.src2))



    def get_resulting_argument(self, argument):
        """ Returns the correct physical argument for a given argument.

        Converts the argument into something MIPS ready:
        3AC register (FP, RV) -> MIPS equivalent
        pseudo-register (ireg_00001) -> $tx
        literal value (100) -> 100

        :param argument:
        :return:
        """

        if isinstance(argument, taci.Register):
            if argument.register in tacr.TAC_REGISTERS:
                return self.convert_tac_register_to_mips_register(argument.register)

            else:
                result = self.register_table.acquire(argument.register)
                self.mips_output.extend(result['code'])
                return result['register']

        elif isinstance(argument, taci.Immediate):
            return argument.value

        elif isinstance(argument, taci.Address):

            if argument.register in tacr.TAC_REGISTERS:
                # this is a little lazy, but it might be good enough, if strange things happen with 3AC, this might be
                # the culprit
                argument.register = self.convert_tac_register_to_mips_register(argument.register)

            elif argument.register:  # if there is one at all                                          <<<-----------------
                # this is a little lazy, but it might be good enough, if strange things happen with 3AC, this might be
                # the culprit
                result = self.register_table.acquire(argument.register)
                self.mips_output.extend(result['code'])
                argument.register = result['register']

            return str(argument)

        elif isinstance(argument, taci.Label):
            return argument.label

        else:
            # TODO: not sure this is an acceptable base case
            # This is super gross and should be fixed by making arguments fall into the above categories
            try:
                return int(argument)
            except:
                result = self.register_table.acquire(argument)
                self.mips_output.extend(result['code'])
                return result['register']

    @staticmethod
    def convert_tac_register_to_mips_register(tac_register):
        if tac_register == tacr.ZERO:
            return mr.ZERO

        elif tac_register == tacr.GP:
            return mr.GP

        elif tac_register == tacr.FP:
            return mr.FP

        elif tac_register == tacr.SP:
            return mr.SP

        elif tac_register == tacr.RA:
            return mr.RA

        elif tac_register == tacr.RV:
            return mr.V0

        else:
            raise Exception('No MIPS equivalent for 3AC register {}'.format(tac_register))
