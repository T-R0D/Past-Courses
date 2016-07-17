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

WORD_SIZE = 4  # aka the size in bytes of a register

ZERO = '$zero'

V0 = '$v0'
V1 = '$v1'

A0 = '$a0'
A1 = '$a1'
A2 = '$a2'
A3 = '$a3'

T0 = '$t0'
T1 = '$t1'
T2 = '$t2'
T3 = '$t3'
T4 = '$t4'
T5 = '$t5'
T6 = '$t6'
T7 = '$t7'
T8 = '$t8'
T9 = '$t9'

S0 = '$s0'
S1 = '$s1'
S2 = '$s2'
S3 = '$s3'
S4 = '$s4'
S5 = '$s5'
S6 = '$s6'
S7 = '$s7'

F2 = '$f2'
F3 = '$f3'
F4 = '$f4'
F5 = '$f5'
F6 = '$f6'
F7 = '$f7'
F8 = '$f8'
F9 = '$f9'
F10 = '$f10'
F11 = '$f11'
F14 = '$f14'
F15 = '$f15'
F16 = '$f16'
F17 = '$f17'
F18 = '$f18'
F19 = '$f19'
F20 = '$f20'
F21 = '$f21'
F22 = '$f22'
F23 = '$f23'
F24 = '$f24'
F25 = '$f25'
F26 = '$f26'
F27 = '$f27'
F28 = '$f28'
F29 = '$f29'
F30 = '$f30'
F31 = '$f31'

# These are used in syscalls involving floats
F0 = '$f0'
F1 = '$f1'
F12 = '$f12'
F13 = '$f13'

GP = '$gp'
SP = '$sp'
FP = '$fp'
RA = '$ra'

PERSISTENT_REGISTERS = (ZERO, GP, SP, FP, RA)

T_REGISTERS = (T0,T1,T2,T3,T4,T5,T6,T7,T8,T9)
S_REGISTERS = (S0, S1, S2, S3, S4, S5, S6, S7)

# these are freely available for use as float registers (or if you are feeling hacky, non-memory storage)
FLOAT_REGISTERS = (F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24,
                   F25, F26, F27, F28, F29, F30, F31)

GENERAL_REGISTERS = (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, S0, S1, S2, S3, S4, S5, S6, S7,)

