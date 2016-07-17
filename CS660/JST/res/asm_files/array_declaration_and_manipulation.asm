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

.data
SPILL_MEMORY: .space 64
.text
add          $fp,      $sp,    $zero
add          $a0,      $fp,    $zero
jal         main
j       PROG_END
main:
CALLEE_FUNCTION_PROLOGUE(40)
li           $t0,        0
mul          $t0,      $t0,        4
la           $t1,    ($fp)
addiu        $t2,      $t1,       12
add          $t0,      $t0,      $t1
tlt          $t0,      $t1
tge          $t0,      $t2
li           $t2,        2
sw           $t2,    ($t0)
li           $t2,        2
mul          $t2,      $t2,        4
la           $t0,    ($fp)
addiu        $t1,      $t0,       12
add          $t2,      $t2,      $t0
tlt          $t2,      $t0
tge          $t2,      $t1
li           $t1,        0
mul          $t1,      $t1,        4
la           $t0,    ($fp)
addiu        $t3,      $t0,       12
add          $t1,      $t1,      $t0
tlt          $t1,      $t0
tge          $t1,      $t3
lw           $t1,    ($t1)
sw           $t1,    ($t2)
la           $t1, -156($fp)
li           $t2,        0
sw           $t2,    ($t1)
LOOP_CONDITION_00003:
lw           $t2, -156($fp)
li           $t1,        3
slt          $t3,      $t2,      $t1
bne          $t3,    $zero, LOOP_BODY_00003
j       LOOP_EXIT_00003
LOOP_BODY_00003:
CALLER_FUNCTION_PROLOGUE()
lw           $t1, -156($fp)
mul          $t1,      $t1,        4
la           $t2,    ($fp)
addiu        $t0,      $t2,       12
add          $t1,      $t1,      $t2
tlt          $t1,      $t2
tge          $t1,      $t0
lw           $t1,    ($t1)
sw           $t1,    ($sp)
sub          $sp,      $sp,        4
jal     print_int
CALLER_FUNCTION_EPILOGUE()
add          $t1,      $v0,    $zero
la           $t1, -156($fp)
lw           $t0,    ($t1)
add          $t2,      $t0,    $zero
addiu        $t0,      $t0,        1
sw           $t0,    ($t1)
j       LOOP_CONDITION_00003
LOOP_EXIT_00003:
li           $t2,        0
mul          $t2,      $t2,        2
li           $t0,        0
addu         $t2,      $t2,      $t0
mul          $t2,      $t2,        4
la           $t0,  12($fp)
addiu        $t1,      $t0,       16
add          $t2,      $t2,      $t0
tlt          $t2,      $t0
tge          $t2,      $t1
li           $t1,       20
sw           $t1,    ($t2)
CALLER_FUNCTION_PROLOGUE()
li           $t1,        0
mul          $t1,      $t1,        2
li           $t2,        0
addu         $t1,      $t1,      $t2
mul          $t1,      $t1,        4
la           $t2,  12($fp)
addiu        $t0,      $t2,       16
add          $t1,      $t1,      $t2
tlt          $t1,      $t2
tge          $t1,      $t0
lw           $t1,    ($t1)
sw           $t1,    ($sp)
sub          $sp,      $sp,        4
jal     print_int
CALLER_FUNCTION_EPILOGUE()
add          $t1,      $v0,    $zero
li           $t1,        1
mul          $t1,      $t1,        2
li           $t0,        1
addu         $t1,      $t1,      $t0
mul          $t1,      $t1,        4
la           $t0,  12($fp)
addiu        $t2,      $t0,       16
add          $t1,      $t1,      $t0
tlt          $t1,      $t0
tge          $t1,      $t2
li           $t2,        0
mul          $t2,      $t2,        2
li           $t0,        0
addu         $t2,      $t2,      $t0
mul          $t2,      $t2,        4
la           $t0,  12($fp)
addiu        $t4,      $t0,       16
add          $t2,      $t2,      $t0
tlt          $t2,      $t0
tge          $t2,      $t4
lw           $t2,    ($t2)
sw           $t2,    ($t1)
CALLER_FUNCTION_PROLOGUE()
li           $t2,        1
mul          $t2,      $t2,        2
li           $t1,        1
addu         $t2,      $t2,      $t1
mul          $t2,      $t2,        4
la           $t1,  12($fp)
addiu        $t4,      $t1,       16
add          $t2,      $t2,      $t1
tlt          $t2,      $t1
tge          $t2,      $t4
lw           $t2,    ($t2)
sw           $t2,    ($sp)
sub          $sp,      $sp,        4
jal     print_int
CALLER_FUNCTION_EPILOGUE()
add          $t2,      $v0,    $zero
li           $t2,        1
mul          $t2,      $t2,        2
li           $t4,        0
addu         $t2,      $t2,      $t4
mul          $t2,      $t2,        2
li           $t4,        1
addu         $t2,      $t2,      $t4
mul          $t2,      $t2,        2
li           $t4,        0
addu         $t2,      $t2,      $t4
mul          $t2,      $t2,        2
li           $t4,        1
addu         $t2,      $t2,      $t4
mul          $t2,      $t2,        4
la           $t4,  28($fp)
addiu        $t1,      $t4,      128
add          $t2,      $t2,      $t4
tlt          $t2,      $t4
tge          $t2,      $t1
li           $t1,       45
sw           $t1,    ($t2)
CALLER_FUNCTION_PROLOGUE()
li           $t1,        1
mul          $t1,      $t1,        2
li           $t2,        0
addu         $t1,      $t1,      $t2
mul          $t1,      $t1,        2
li           $t2,        1
addu         $t1,      $t1,      $t2
mul          $t1,      $t1,        2
li           $t2,        0
addu         $t1,      $t1,      $t2
mul          $t1,      $t1,        2
li           $t2,        1
addu         $t1,      $t1,      $t2
mul          $t1,      $t1,        4
la           $t2,  28($fp)
addiu        $t4,      $t2,      128
add          $t1,      $t1,      $t2
tlt          $t1,      $t2
tge          $t1,      $t4
lw           $t1,    ($t1)
sw           $t1,    ($sp)
sub          $sp,      $sp,        4
jal     print_int
CALLER_FUNCTION_EPILOGUE()
add          $t1,      $v0,    $zero
li           $t1,        0
mul          $t1,      $t1,        2
li           $t4,        0
addu         $t1,      $t1,      $t4
mul          $t1,      $t1,        2
li           $t4,        0
addu         $t1,      $t1,      $t4
mul          $t1,      $t1,        2
li           $t4,        0
addu         $t1,      $t1,      $t4
mul          $t1,      $t1,        2
li           $t4,        1
addu         $t1,      $t1,      $t4
mul          $t1,      $t1,        4
la           $t4,  28($fp)
addiu        $t2,      $t4,      128
add          $t1,      $t1,      $t4
tlt          $t1,      $t4
tge          $t1,      $t2
li           $t2,        1
mul          $t2,      $t2,        2
li           $t4,        0
addu         $t2,      $t2,      $t4
mul          $t2,      $t2,        2
li           $t4,        1
addu         $t2,      $t2,      $t4
mul          $t2,      $t2,        2
li           $t4,        0
addu         $t2,      $t2,      $t4
mul          $t2,      $t2,        2
li           $t4,        1
addu         $t2,      $t2,      $t4
mul          $t2,      $t2,        4
la           $t4,  28($fp)
addiu        $t0,      $t4,      128
add          $t2,      $t2,      $t4
tlt          $t2,      $t4
tge          $t2,      $t0
lw           $t2,    ($t2)
sw           $t2,    ($t1)
CALLER_FUNCTION_PROLOGUE()
li           $t2,        0
mul          $t2,      $t2,        2
li           $t1,        0
addu         $t2,      $t2,      $t1
mul          $t2,      $t2,        2
li           $t1,        0
addu         $t2,      $t2,      $t1
mul          $t2,      $t2,        2
li           $t1,        0
addu         $t2,      $t2,      $t1
mul          $t2,      $t2,        2
li           $t1,        1
addu         $t2,      $t2,      $t1
mul          $t2,      $t2,        4
la           $t1,  28($fp)
addiu        $t0,      $t1,      128
add          $t2,      $t2,      $t1
tlt          $t2,      $t1
tge          $t2,      $t0
lw           $t2,    ($t2)
sw           $t2,    ($sp)
sub          $sp,      $sp,        4
jal     print_int
CALLER_FUNCTION_EPILOGUE()
add          $t2,      $v0,    $zero
li           $t2,        0
add          $v0,      $t2,    $zero
CALLEE_FUNCTION_EPILOGUE()
CALLEE_FUNCTION_EPILOGUE()
print_int:
CALLEE_FUNCTION_PROLOGUE(0)
# load $v0 with the value for the print int syscall
li           $v0,        1
# the first (and only) argument is the value to print
lw           $a0,    ($fp)
syscall 
# print a newline character for readability
# 0x0D is CR or '\r' - 0x0A is LF for '\n'
li           $v0,       11
li           $a0,       10
syscall 
CALLEE_FUNCTION_EPILOGUE()
print_string:
CALLEE_FUNCTION_PROLOGUE(0)
# load $v0 with the value for the print int syscall
li           $v0,        4
# the first (and only) argument is the base address of the null terminated ascii string
la           $a0,    ($fp)
syscall 
CALLEE_FUNCTION_EPILOGUE()
print_float:
CALLEE_FUNCTION_PROLOGUE(0)
# load $v0 with the value for the print int syscall
li           $v0,        2
# the first (and only) argument is the base address of the null terminated ascii string
lwc1        $f12,    ($fp)
syscall 
CALLEE_FUNCTION_EPILOGUE()
PROG_END: