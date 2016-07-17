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
beqz        %lhs, __LOR_FALSE
beqz        %rhs, __LOR_FALSE
li           $a2,        0
j       __LOR_END

__LOR_FALSE:
li           $a2,        1

__LOR_END:
.end_macro


.data
SPILL_MEMORY: .space 64

.text
add          $fp,      $sp,    $zero
add          $a0,      $fp,    $zero
jal         main
j       PROG_END
## SOURCE ## 0: 
## SOURCE ## 1:             int foo(int a[][])

foo:
CALLEE_FUNCTION_PROLOGUE(2)
## SOURCE ## 2:             {
## SOURCE ## 3:                 int i, j;
## SOURCE ## 4:                 for(i = 0; i < 3; ++i)
la           $t0, -16($fp)
li           $t1,        0
sw           $t1,    ($t0)

LOOP_CONDITION_00000:
lw           $t1, -16($fp)
li           $t0,        3
slt          $t2,      $t1,      $t0
bne          $t2,    $zero, LOOP_BODY_00000
j       LOOP_EXIT_00000

LOOP_BODY_00000:
## SOURCE ## 5:                 {
## SOURCE ## 6:                     for(j = 0; j < 7; ++j)
la           $t0, -20($fp)
li           $t1,        0
sw           $t1,    ($t0)

LOOP_CONDITION_00001:
lw           $t1, -20($fp)
li           $t0,        7
slt          $t3,      $t1,      $t0
bne          $t3,    $zero, LOOP_BODY_00001
j       LOOP_EXIT_00001

LOOP_BODY_00001:
## SOURCE ## 7:                     {
CALLER_FUNCTION_PROLOGUE()
## SOURCE ## 8:                         print_char(i + '0');
lw           $t0, -16($fp)
li           $t1,       48
add          $t4,      $t0,      $t1
sw           $t4,    ($sp)
sub          $sp,      $sp,        4
jal     print_char
CALLER_FUNCTION_EPILOGUE()
add          $t4,      $v0,    $zero
CALLER_FUNCTION_PROLOGUE()
## SOURCE ## 9:                         print_char(' ');
li           $t4,       32
sw           $t4,    ($sp)
sub          $sp,      $sp,        4
jal     print_char
CALLER_FUNCTION_EPILOGUE()
add          $t4,      $v0,    $zero
CALLER_FUNCTION_PROLOGUE()
## SOURCE ## 10:                         print_char(j + '0');
lw           $t4, -20($fp)
li           $t1,       48
add          $t0,      $t4,      $t1
sw           $t0,    ($sp)
sub          $sp,      $sp,        4
jal     print_char
CALLER_FUNCTION_EPILOGUE()
add          $t0,      $v0,    $zero
CALLER_FUNCTION_PROLOGUE()
## SOURCE ## 11:                         print_char(' ');
li           $t0,       32
sw           $t0,    ($sp)
sub          $sp,      $sp,        4
jal     print_char
CALLER_FUNCTION_EPILOGUE()
add          $t0,      $v0,    $zero
CALLER_FUNCTION_PROLOGUE()
## SOURCE ## 12:                         print_char(' ');
li           $t0,       32
sw           $t0,    ($sp)
sub          $sp,      $sp,        4
jal     print_char
CALLER_FUNCTION_EPILOGUE()
add          $t0,      $v0,    $zero
CALLER_FUNCTION_PROLOGUE()
## SOURCE ## 13:                         print_char(' ');
li           $t0,       32
sw           $t0,    ($sp)
sub          $sp,      $sp,        4
jal     print_char
CALLER_FUNCTION_EPILOGUE()
add          $t0,      $v0,    $zero
CALLER_FUNCTION_PROLOGUE()
## SOURCE ## 14:                         print_int(a[i][j]);
lw           $t0, -16($fp)
lw           $t1, -12($fp)
mul          $t0,      $t0,      $t1
lw           $t1, -20($fp)
addu         $t0,      $t0,      $t1
mul          $t0,      $t0,        4
lw           $t1,    ($fp)
lw           $t4,  -4($fp)
sub          $t4,      $t1,      $t4
sub          $t0,      $t1,      $t0
tlt          $t1,      $t0
tge          $t4,      $t0
lw           $t0,    ($t0)
sw           $t0,    ($sp)
sub          $sp,      $sp,        4
jal     print_int
CALLER_FUNCTION_EPILOGUE()
add          $t0,      $v0,    $zero
CALLER_FUNCTION_PROLOGUE()
## SOURCE ## 15:                         print_char('\n');
li           $t0,       10
sw           $t0,    ($sp)
sub          $sp,      $sp,        4
jal     print_char
CALLER_FUNCTION_EPILOGUE()
add          $t0,      $v0,    $zero
la           $t0, -20($fp)
lw           $t4,    ($t0)
addiu        $t4,      $t4,        1
sw           $t4,    ($t0)
j       LOOP_CONDITION_00001

LOOP_EXIT_00001:
la           $t4, -16($fp)
lw           $t0,    ($t4)
addiu        $t0,      $t0,        1
sw           $t0,    ($t4)
j       LOOP_CONDITION_00000

LOOP_EXIT_00000:
CALLEE_FUNCTION_EPILOGUE()
## SOURCE ## 16:                     }
## SOURCE ## 17:                 }
## SOURCE ## 18:             }
## SOURCE ## 19: 
## SOURCE ## 20:             int main() {

main:
CALLEE_FUNCTION_PROLOGUE(23)
## SOURCE ## 21: 
## SOURCE ## 22:                 int b[3][7];
## SOURCE ## 23:                 int i, j;
## SOURCE ## 24:                 for(i = 0; i < 3; ++i)
la           $t2, -84($fp)
li           $t3,        0
sw           $t3,    ($t2)

LOOP_CONDITION_00002:
lw           $t3, -84($fp)
li           $t2,        3
slt          $t0,      $t3,      $t2
bne          $t0,    $zero, LOOP_BODY_00002
j       LOOP_EXIT_00002

LOOP_BODY_00002:
## SOURCE ## 25:                 {
## SOURCE ## 26:                     for(j = 0; j < 7; ++j)
la           $t2, -88($fp)
li           $t3,        0
sw           $t3,    ($t2)

LOOP_CONDITION_00003:
lw           $t3, -88($fp)
li           $t2,        7
slt          $t4,      $t3,      $t2
bne          $t4,    $zero, LOOP_BODY_00003
j       LOOP_EXIT_00003

LOOP_BODY_00003:
## SOURCE ## 27:                     {
## SOURCE ## 28:                         b[i][j] = (i*7) + j;
lw           $t2, -84($fp)
mul          $t2,      $t2,        7
lw           $t3, -88($fp)
addu         $t2,      $t2,      $t3
mul          $t2,      $t2,        4
la           $t3,    ($fp)
addi         $t1,      $t3,      -84
sub          $t2,      $t3,      $t2
tlt          $t3,      $t2
tge          $t1,      $t2
lw           $t1, -84($fp)
li           $t3,        7
mul          $t5,      $t1,      $t3
lw           $t3, -88($fp)
add          $t1,      $t5,      $t3
sw           $t1,    ($t2)
la           $t1, -88($fp)
lw           $t2,    ($t1)
addiu        $t2,      $t2,        1
sw           $t2,    ($t1)
j       LOOP_CONDITION_00003

LOOP_EXIT_00003:
la           $t2, -84($fp)
lw           $t1,    ($t2)
addiu        $t1,      $t1,        1
sw           $t1,    ($t2)
j       LOOP_CONDITION_00002

LOOP_EXIT_00002:
CALLER_FUNCTION_PROLOGUE()
lw           $t1,    ($fp)
la           $t1,    ($fp)
sw           $t1,    ($sp)
sub          $sp,      $sp,        4
li           $t1,       84
sw           $t1,    ($sp)
sub          $sp,      $sp,        4
li           $t1,        3
sw           $t1,    ($sp)
sub          $sp,      $sp,        4
li           $t1,        7
sw           $t1,    ($sp)
sub          $sp,      $sp,        4
jal          foo
CALLER_FUNCTION_EPILOGUE()
add          $t1,      $v0,    $zero
## SOURCE ## 29:                     }
## SOURCE ## 30:                 }
## SOURCE ## 31: 
## SOURCE ## 32:                 foo(b);
## SOURCE ## 33: 
## SOURCE ## 34:                 return 0;
li           $t1,        0
add          $v0,      $t1,    $zero
CALLEE_FUNCTION_EPILOGUE()
CALLEE_FUNCTION_EPILOGUE()
## SOURCE ## 35:             }
## SOURCE ## 36:             

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