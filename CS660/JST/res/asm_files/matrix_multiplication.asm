.macro SAVE_T_REGISTERS
# brace yourself for a long, unrolled loop...
sw           $t0,    ($sp)
subiu        $sp,      $sp,        4
sw           $t1,    ($sp)
subiu        $sp,      $sp,        4
sw           $t2,    ($sp)
subiu        $sp,      $sp,        4
sw           $t3,    ($sp)
subiu        $sp,      $sp,        4
sw           $t4,    ($sp)
subiu        $sp,      $sp,        4
sw           $t5,    ($sp)
subiu        $sp,      $sp,        4
sw           $t6,    ($sp)
subiu        $sp,      $sp,        4
sw           $t7,    ($sp)
subiu        $sp,      $sp,        4
sw           $t8,    ($sp)
subiu        $sp,      $sp,        4
sw           $t9,    ($sp)
subiu        $sp,      $sp,        4
.end_macro

.macro RESTORE_T_REGISTERS
# brace yourself for a long, unrolled loop...
addiu        $sp,      $sp,        4
lw           $t9,    ($sp)
addiu        $sp,      $sp,        4
lw           $t8,    ($sp)
addiu        $sp,      $sp,        4
lw           $t7,    ($sp)
addiu        $sp,      $sp,        4
lw           $t6,    ($sp)
addiu        $sp,      $sp,        4
lw           $t5,    ($sp)
addiu        $sp,      $sp,        4
lw           $t4,    ($sp)
addiu        $sp,      $sp,        4
lw           $t3,    ($sp)
addiu        $sp,      $sp,        4
lw           $t2,    ($sp)
addiu        $sp,      $sp,        4
lw           $t1,    ($sp)
addiu        $sp,      $sp,        4
lw           $t0,    ($sp)
.end_macro

.macro SAVE_SPILL_MEM
# brace yourself for a long, unrolled loop...
lw           $a3, SPILL_MEMORY
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 4
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 8
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 12
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 16
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 20
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 24
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 28
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 32
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 36
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 40
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 44
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 48
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 52
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 56
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
lw           $a3, SPILL_MEMORY + 60
sw           $a3,    ($sp)
subiu        $sp,      $sp,        4
.end_macro

.macro RESTORE_SPILL_MEM
# brace yourself for a long, unrolled loop...
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 60
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 56
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 52
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 48
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 44
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 40
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 36
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 32
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 28
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 24
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 20
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 16
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 12
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 8
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY + 4
addiu        $sp,      $sp,        4
lw           $a3,    ($sp)
sw           $a3, SPILL_MEMORY
.end_macro

.macro CALLEE_FUNCTION_PROLOGUE (%variable_size)
# set $fp to the proper spot by recovering the value from $a0
add          $fp,      $a0,    $zero
# allocate stack space for variables ($sp = $sp - space for variables)
li           $a0,        4
mulu         $a1,      $a0, %variable_size
sub          $sp,      $sp,      $a1
.end_macro

.macro CALLEE_FUNCTION_EPILOGUE
# de-allocate the memory used for local variables and parameters
add          $sp,      $fp,    $zero
# jump back to the caller
jr           $ra
.end_macro

.macro CALLER_FUNCTION_PROLOGUE
# caller should save it's own $ra, $fp, and registers
sw           $ra,    ($sp)
subiu        $sp,      $sp,        4
sw           $fp,    ($sp)
subiu        $sp,      $sp,        4
# caller pushes registers and spill memory onto the stack as well
SAVE_T_REGISTERS()
SAVE_SPILL_MEM()
# save the value of $sp here into $a0 as temporary storage until the arguments are moved
# $fp needs to stay where it's at while the arguments are copied after this macro
add          $a0,      $sp,    $zero
.end_macro

.macro CALLER_FUNCTION_EPILOGUE
# recover the spill memory and the stored registers
RESTORE_SPILL_MEM()
RESTORE_T_REGISTERS()
# recover the caller's $fp and $ra
addiu        $sp,      $sp,        4
lw           $fp,    ($sp)
addiu        $sp,      $sp,        4
lw           $ra,    ($sp)
.end_macro

.macro __LAND (%lhs, %rhs)
beqz        %lhs, __LAND_FALSE
beqz        %rhs, __LAND_FALSE
li           $a2,        1
j       __LAND_END

__LAND_FALSE:
li           $a2,        0

__LAND_END:
.end_macro

.macro __LOR (%lhs, %rhs)
beqz        %lhs, __LOR_TRUE
beqz        %rhs, __LOR_TRUE
li           $a2,        0
j       __LOR_END

__LOR_TRUE:
li           $a2,        1

__LOR_END:
.end_macro


.data
SPILL_MEMORY: .space 64
ARRAY_DIM: .word 2

.text
add          $fp,      $sp,    $zero
add          $a0,      $fp,    $zero
jal         main
j       PROG_END

main:
CALLEE_FUNCTION_PROLOGUE(16)
la           $t0,    ($fp)
li           $t1,        0
sw           $t1,    ($t0)

LOOP_CONDITION_00000:
lw           $t1,    ($fp)
li           $t0,        2
lw           $t0, ARRAY_DIM
slt          $t2,      $t1,      $t0
bne          $t2,    $zero, LOOP_BODY_00000
j       LOOP_EXIT_00000

LOOP_BODY_00000:
la           $t0,  -4($fp)
li           $t1,        0
sw           $t1,    ($t0)

LOOP_CONDITION_00001:
lw           $t1,  -4($fp)
li           $t0,        2
lw           $t0, ARRAY_DIM
slt          $t3,      $t1,      $t0
bne          $t3,    $zero, LOOP_BODY_00001
j       LOOP_EXIT_00001

LOOP_BODY_00001:
lw           $t0,    ($fp)
mul          $t0,      $t0,        2
lw           $t1,  -4($fp)
addu         $t0,      $t0,      $t1
mul          $t0,      $t0,        4
la           $t1, -16($fp)
addi         $t4,      $t1,      -16
sub          $t0,      $t1,      $t0
tlt          $t1,      $t0
tge          $t4,      $t0
lw           $t4,    ($fp)
mul          $t4,      $t4,        2
lw           $t1,  -4($fp)
addu         $t4,      $t4,      $t1
mul          $t4,      $t4,        4
la           $t1, -32($fp)
addi         $t5,      $t1,      -16
sub          $t4,      $t1,      $t4
tlt          $t1,      $t4
tge          $t5,      $t4
li           $t5,        2
sw           $t5,    ($t4)
sw           $t5,    ($t0)
la           $t5,  -4($fp)
lw           $t0,    ($t5)
add          $t4,      $t0,    $zero
addiu        $t0,      $t0,        1
sw           $t0,    ($t5)
j       LOOP_CONDITION_00001

LOOP_EXIT_00001:
la           $t4,    ($fp)
lw           $t0,    ($t4)
add          $t5,      $t0,    $zero
addiu        $t0,      $t0,        1
sw           $t0,    ($t4)
j       LOOP_CONDITION_00000

LOOP_EXIT_00000:
CALLER_FUNCTION_PROLOGUE()
lw           $t5, -48($fp)
la           $t5,    ($fp)
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
li           $t5,       16
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
li           $t5,        2
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
li           $t5,        2
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
lw           $t5, -16($fp)
la           $t5, -16($fp)
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
li           $t5,       16
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
li           $t5,        2
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
li           $t5,        2
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
lw           $t5, -32($fp)
la           $t5, -32($fp)
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
li           $t5,       16
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
li           $t5,        2
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
li           $t5,        2
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
jal     matrix_multiply
CALLER_FUNCTION_EPILOGUE()
add          $t5,      $v0,    $zero
CALLER_FUNCTION_PROLOGUE()
lw           $t5, -48($fp)
la           $t5,    ($fp)
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
li           $t5,       16
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
li           $t5,        2
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
li           $t5,        2
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
jal     print_matrix
CALLER_FUNCTION_EPILOGUE()
add          $t5,      $v0,    $zero
li           $t5,        0
add          $v0,      $t5,    $zero
CALLEE_FUNCTION_EPILOGUE()
CALLEE_FUNCTION_EPILOGUE()

matrix_multiply:
CALLEE_FUNCTION_PROLOGUE(4)
la           $t2, -48($fp)
li           $t3,        0
sw           $t3,    ($t2)

LOOP_CONDITION_00002:
lw           $t3, -48($fp)
li           $t2,        2
lw           $t2, ARRAY_DIM
slt          $t5,      $t3,      $t2
bne          $t5,    $zero, LOOP_BODY_00002
j       LOOP_EXIT_00002

LOOP_BODY_00002:
la           $t2, -52($fp)
li           $t3,        0
sw           $t3,    ($t2)

LOOP_CONDITION_00003:
lw           $t3, -52($fp)
li           $t2,        2
lw           $t2, ARRAY_DIM
slt          $t0,      $t3,      $t2
bne          $t0,    $zero, LOOP_BODY_00003
j       LOOP_EXIT_00003

LOOP_BODY_00003:
la           $t2, -60($fp)
li           $t3,        0
sw           $t3,    ($t2)
la           $t3, -56($fp)
li           $t2,        0
sw           $t2,    ($t3)

LOOP_CONDITION_00004:
lw           $t2, -56($fp)
li           $t3,        2
lw           $t3, ARRAY_DIM
slt          $t4,      $t2,      $t3
bne          $t4,    $zero, LOOP_BODY_00004
j       LOOP_EXIT_00004

LOOP_BODY_00004:
la           $t3, -60($fp)
lw           $t2, -60($fp)
lw           $t1, -48($fp)
lw           $t6, -28($fp)
mul          $t1,      $t1,      $t6
lw           $t6, -56($fp)
addu         $t1,      $t1,      $t6
mul          $t1,      $t1,        4
lw           $t6, -16($fp)
lw           $t7, -20($fp)
sub          $t7,      $t6,      $t7
sub          $t1,      $t6,      $t1
tlt          $t6,      $t1
tge          $t7,      $t1
lw           $t1,    ($t1)
lw           $t7, -56($fp)
lw           $t6, -44($fp)
mul          $t7,      $t7,      $t6
lw           $t6, -52($fp)
addu         $t7,      $t7,      $t6
mul          $t7,      $t7,        4
lw           $t6, -32($fp)
lw           $t8, -36($fp)
sub          $t8,      $t6,      $t8
sub          $t7,      $t6,      $t7
tlt          $t6,      $t7
tge          $t8,      $t7
lw           $t7,    ($t7)
mul          $t8,      $t1,      $t7
add          $t7,      $t2,      $t8
sw           $t7,    ($t3)
la           $t7, -56($fp)
lw           $t3,    ($t7)
add          $t8,      $t3,    $zero
addiu        $t3,      $t3,        1
sw           $t3,    ($t7)
j       LOOP_CONDITION_00004

LOOP_EXIT_00004:
lw           $t8, -48($fp)
lw           $t3, -12($fp)
mul          $t8,      $t8,      $t3
lw           $t3, -52($fp)
addu         $t8,      $t8,      $t3
mul          $t8,      $t8,        4
lw           $t3,    ($fp)
lw           $t7,  -4($fp)
sub          $t7,      $t3,      $t7
sub          $t8,      $t3,      $t8
tlt          $t3,      $t8
tge          $t7,      $t8
lw           $t7, -60($fp)
sw           $t7,    ($t8)
la           $t7, -52($fp)
lw           $t8,    ($t7)
add          $t3,      $t8,    $zero
addiu        $t8,      $t8,        1
sw           $t8,    ($t7)
j       LOOP_CONDITION_00003

LOOP_EXIT_00003:
la           $t3, -48($fp)
lw           $t8,    ($t3)
add          $t7,      $t8,    $zero
addiu        $t8,      $t8,        1
sw           $t8,    ($t3)
j       LOOP_CONDITION_00002

LOOP_EXIT_00002:
CALLEE_FUNCTION_EPILOGUE()

print_matrix:
CALLEE_FUNCTION_PROLOGUE(2)
la           $t5, -16($fp)
li           $t0,        0
sw           $t0,    ($t5)

LOOP_CONDITION_00005:
lw           $t0, -16($fp)
li           $t5,        2
lw           $t5, ARRAY_DIM
slt          $t4,      $t0,      $t5
bne          $t4,    $zero, LOOP_BODY_00005
j       LOOP_EXIT_00005

LOOP_BODY_00005:
la           $t5, -20($fp)
li           $t0,        0
sw           $t0,    ($t5)

LOOP_CONDITION_00006:
lw           $t0, -20($fp)
li           $t5,        2
lw           $t5, ARRAY_DIM
slt          $t7,      $t0,      $t5
bne          $t7,    $zero, LOOP_BODY_00006
j       LOOP_EXIT_00006

LOOP_BODY_00006:
CALLER_FUNCTION_PROLOGUE()
lw           $t5, -16($fp)
lw           $t0, -12($fp)
mul          $t5,      $t5,      $t0
lw           $t0, -20($fp)
addu         $t5,      $t5,      $t0
mul          $t5,      $t5,        4
lw           $t0,    ($fp)
lw           $t8,  -4($fp)
sub          $t8,      $t0,      $t8
sub          $t5,      $t0,      $t5
tlt          $t0,      $t5
tge          $t8,      $t5
lw           $t5,    ($t5)
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
jal     print_int
CALLER_FUNCTION_EPILOGUE()
add          $t5,      $v0,    $zero
CALLER_FUNCTION_PROLOGUE()
li           $t5,       32
sw           $t5,    ($sp)
sub          $sp,      $sp,        4
jal     print_char
CALLER_FUNCTION_EPILOGUE()
add          $t5,      $v0,    $zero
la           $t5, -20($fp)
lw           $t8,    ($t5)
add          $t0,      $t8,    $zero
addiu        $t8,      $t8,        1
sw           $t8,    ($t5)
j       LOOP_CONDITION_00006

LOOP_EXIT_00006:
CALLER_FUNCTION_PROLOGUE()
li           $t0,       10
sw           $t0,    ($sp)
sub          $sp,      $sp,        4
jal     print_char
CALLER_FUNCTION_EPILOGUE()
add          $t0,      $v0,    $zero
la           $t0, -16($fp)
lw           $t8,    ($t0)
add          $t5,      $t8,    $zero
addiu        $t8,      $t8,        1
sw           $t8,    ($t0)
j       LOOP_CONDITION_00005

LOOP_EXIT_00005:
CALLEE_FUNCTION_EPILOGUE()

print_char:
CALLEE_FUNCTION_PROLOGUE(0)
# load $v0 with the value for the print char syscall
li           $v0,       11
# the first (and only) argument is the value to print
lw           $a0,    ($fp)
syscall 
CALLEE_FUNCTION_EPILOGUE()

print_int:
CALLEE_FUNCTION_PROLOGUE(0)
# load $v0 with the value for the print int syscall
li           $v0,        1
# the first (and only) argument is the value to print
lw           $a0,    ($fp)
syscall 
CALLEE_FUNCTION_EPILOGUE()

print_string:
CALLEE_FUNCTION_PROLOGUE(0)
# load $v0 with the value for the print string syscall
li           $v0,        4
# the first (and only) argument is the base address of the null terminated ascii string
la           $a0,    ($fp)
syscall 
CALLEE_FUNCTION_EPILOGUE()

print_float:
CALLEE_FUNCTION_PROLOGUE(0)
# load $v0 with the value for the print float syscall
li           $v0,        2
# the first (and only) argument is the base address of the null terminated ascii string
lwc1        $f12,    ($fp)
syscall 
CALLEE_FUNCTION_EPILOGUE()

PROG_END:
add          $a0,      $v0,    $zero
li           $v0,       17
syscall 