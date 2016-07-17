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

##  3AC   ## .data

.data
SPILL_MEMORY: .space 64
##  3AC   ## .text

.text
add          $fp,      $sp,    $zero
add          $a0,      $fp,    $zero
##  3AC   ## JAL            , main           , -              , -              
jal         main
##  3AC   ## BR             , PROG_END       , -              , -              
j       PROG_END
## SOURCE ## 0: 
## SOURCE ## 1:             int main() {
##  3AC   ## main:

main:
##  3AC   ## PROCENTRY      , 4              , -              , -              
CALLEE_FUNCTION_PROLOGUE(1)
## SOURCE ## 2:                 int i = 0;
##  3AC   ## LA             , ireg_00000     , ($FP)          , -              
la           $t0,    ($fp)
##  3AC   ## LI             , ireg_00001     , 0              , -              
li           $t1,        0
##  3AC   ## STORE          , ireg_00001     , (ireg_00000)   , 4              
sw           $t1,    ($t0)
##  3AC   ## KICK           , ireg_00001     , -              , -              
##  3AC   ## KICK           , ireg_00000     , -              , -              
## SOURCE ## 3: 
## SOURCE ## 4:                 // FizzBuzz
## SOURCE ## 5:                 for( i = 1; i <= 30; i++) {
##  3AC   ## LA             , ireg_00002     , ($FP)          , -              
la           $t0,    ($fp)
##  3AC   ## LI             , ireg_00003     , 1              , -              
li           $t1,        1
##  3AC   ## STORE          , ireg_00003     , (ireg_00002)   , 4              
sw           $t1,    ($t0)
##  3AC   ## KICK           , ireg_00002     , -              , -              
##  3AC   ## KICK           , ireg_00003     , -              , -              
##  3AC   ## LOOP_CONDITION_00000:

LOOP_CONDITION_00000:
##  3AC   ## LOAD           , ireg_00004     , ($FP)          , 4              
lw           $t1,    ($fp)
##  3AC   ## LI             , ireg_00005     , 30             , -              
li           $t0,       30
##  3AC   ## LE             , ireg_00006     , ireg_00004     , ireg_00005     
sle          $t2,      $t1,      $t0
##  3AC   ## KICK           , ireg_00004     , -              , -              
##  3AC   ## KICK           , ireg_00005     , -              , -              
##  3AC   ## BRNE           , LOOP_BODY_00000, ireg_00006     , $ZERO          
bne          $t2,    $zero, LOOP_BODY_00000
##  3AC   ## BR             , LOOP_EXIT_00000, -              , -              
j       LOOP_EXIT_00000
##  3AC   ## LOOP_BODY_00000:

LOOP_BODY_00000:
## SOURCE ## 6:                    //FizzBuzz
## SOURCE ## 7:                    if( i % 3 == 0){
##  3AC   ## LOAD           , ireg_00007     , ($FP)          , 4              
lw           $t0,    ($fp)
##  3AC   ## LI             , ireg_00008     , 3              , -              
li           $t1,        3
##  3AC   ## MOD            , ireg_00009     , ireg_00007     , ireg_00008     
DIV          $t0,$t1
MFHI         $t3
##  3AC   ## KICK           , ireg_00007     , -              , -              
##  3AC   ## KICK           , ireg_00008     , -              , -              
##  3AC   ## LI             , ireg_00010     , 0              , -              
li           $t1,        0
##  3AC   ## EQ             , ireg_00011     , ireg_00009     , ireg_00010     
seq          $t0,      $t3,      $t1
##  3AC   ## KICK           , ireg_00009     , -              , -              
##  3AC   ## KICK           , ireg_00010     , -              , -              
##  3AC   ## BRNE           , IF_TRUE_00000  , ireg_00011     , $ZERO          
bne          $t0,    $zero, IF_TRUE_00000
##  3AC   ## IF_FALSE_00000:

IF_FALSE_00000:
## SOURCE ## 8: 
## SOURCE ## 9:                         //FizzBuzz
## SOURCE ## 10:                         if( i % 5 == 0 ){
## SOURCE ## 11:                             // expect to see this at 15 and 30
## SOURCE ## 12:                             print_int(i); print_char(':'); print_char(' ');
## SOURCE ## 13:                             print_char('f'); print_char('b');
## SOURCE ## 14:                             print_char('\n');
## SOURCE ## 15:                         }
## SOURCE ## 16: 
## SOURCE ## 17:                         //Fizz
## SOURCE ## 18:                         else {
## SOURCE ## 19:                            // expect to see this at 3,6,9,12,18,21,24,27
## SOURCE ## 20:                            print_int(i); print_char(':'); print_char(' ');
## SOURCE ## 21:                            print_char('f');
## SOURCE ## 22:                            print_char('\n');
## SOURCE ## 23:                         }
## SOURCE ## 24: 
## SOURCE ## 25:                    }
## SOURCE ## 26:                    // Buzz
## SOURCE ## 27:                    else if( i % 5 == 0) {
##  3AC   ## LOAD           , ireg_00012     , ($FP)          , 4              
lw           $t1,    ($fp)
##  3AC   ## LI             , ireg_00013     , 5              , -              
li           $t3,        5
##  3AC   ## MOD            , ireg_00014     , ireg_00012     , ireg_00013     
DIV          $t1,$t3
MFHI         $t4
##  3AC   ## KICK           , ireg_00012     , -              , -              
##  3AC   ## KICK           , ireg_00013     , -              , -              
##  3AC   ## LI             , ireg_00015     , 0              , -              
li           $t3,        0
##  3AC   ## EQ             , ireg_00016     , ireg_00014     , ireg_00015     
seq          $t1,      $t4,      $t3
##  3AC   ## KICK           , ireg_00014     , -              , -              
##  3AC   ## KICK           , ireg_00015     , -              , -              
##  3AC   ## BRNE           , IF_TRUE_00001  , ireg_00016     , $ZERO          
bne          $t1,    $zero, IF_TRUE_00001
##  3AC   ## IF_FALSE_00001:

IF_FALSE_00001:
## SOURCE ## 28:                        // expect to see this at 5,10,15,20,25
## SOURCE ## 29:                        print_int(i); print_char(':'); print_char(' ');
## SOURCE ## 30:                        print_char('b');
## SOURCE ## 31:                        print_char('\n');
## SOURCE ## 32:                    }
## SOURCE ## 33:                    // Number
## SOURCE ## 34:                    else {
##  3AC   ## CALL_PROC      , print_int      , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LOAD           , ireg_00018     , ($FP)          , 4              
lw           $t3,    ($fp)
##  3AC   ## STORE          , ireg_00018     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00018     , -              , -              
##  3AC   ## JAL            , print_int      , -              , -              
jal     print_int
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00017     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00017     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
## SOURCE ## 35:                        // expect to see all other numbers except those mentioned above
## SOURCE ## 36:                        print_int(i); print_char('\n');
##  3AC   ## LI             , ireg_00020     , 10             , -              
li           $t3,       10
##  3AC   ## STORE          , ireg_00020     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00020     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00019     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00019     , -              , -              
##  3AC   ## BR             , ENDIF_00001    , -              , -              
j       ENDIF_00001
##  3AC   ## IF_TRUE_00001:

IF_TRUE_00001:
##  3AC   ## CALL_PROC      , print_int      , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LOAD           , ireg_00022     , ($FP)          , 4              
lw           $t3,    ($fp)
##  3AC   ## STORE          , ireg_00022     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00022     , -              , -              
##  3AC   ## JAL            , print_int      , -              , -              
jal     print_int
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00021     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00021     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LI             , ireg_00024     , 58             , -              
li           $t3,       58
##  3AC   ## STORE          , ireg_00024     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00024     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00023     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00023     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LI             , ireg_00026     , 32             , -              
li           $t3,       32
##  3AC   ## STORE          , ireg_00026     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00026     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00025     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00025     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LI             , ireg_00028     , 98             , -              
li           $t3,       98
##  3AC   ## STORE          , ireg_00028     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00028     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00027     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00027     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LI             , ireg_00030     , 10             , -              
li           $t3,       10
##  3AC   ## STORE          , ireg_00030     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00030     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00029     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00029     , -              , -              
##  3AC   ## ENDIF_00001:

ENDIF_00001:
##  3AC   ## KICK           , ireg_00016     , -              , -              
##  3AC   ## BR             , ENDIF_00000    , -              , -              
j       ENDIF_00000
##  3AC   ## IF_TRUE_00000:

IF_TRUE_00000:
##  3AC   ## LOAD           , ireg_00031     , ($FP)          , 4              
lw           $t1,    ($fp)
##  3AC   ## LI             , ireg_00032     , 5              , -              
li           $t3,        5
##  3AC   ## MOD            , ireg_00033     , ireg_00031     , ireg_00032     
DIV          $t1,$t3
MFHI         $t4
##  3AC   ## KICK           , ireg_00031     , -              , -              
##  3AC   ## KICK           , ireg_00032     , -              , -              
##  3AC   ## LI             , ireg_00034     , 0              , -              
li           $t3,        0
##  3AC   ## EQ             , ireg_00035     , ireg_00033     , ireg_00034     
seq          $t1,      $t4,      $t3
##  3AC   ## KICK           , ireg_00033     , -              , -              
##  3AC   ## KICK           , ireg_00034     , -              , -              
##  3AC   ## BRNE           , IF_TRUE_00002  , ireg_00035     , $ZERO          
bne          $t1,    $zero, IF_TRUE_00002
##  3AC   ## IF_FALSE_00002:

IF_FALSE_00002:
##  3AC   ## CALL_PROC      , print_int      , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LOAD           , ireg_00037     , ($FP)          , 4              
lw           $t3,    ($fp)
##  3AC   ## STORE          , ireg_00037     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00037     , -              , -              
##  3AC   ## JAL            , print_int      , -              , -              
jal     print_int
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00036     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00036     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LI             , ireg_00039     , 58             , -              
li           $t3,       58
##  3AC   ## STORE          , ireg_00039     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00039     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00038     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00038     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LI             , ireg_00041     , 32             , -              
li           $t3,       32
##  3AC   ## STORE          , ireg_00041     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00041     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00040     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00040     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LI             , ireg_00043     , 102            , -              
li           $t3,      102
##  3AC   ## STORE          , ireg_00043     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00043     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00042     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00042     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LI             , ireg_00045     , 10             , -              
li           $t3,       10
##  3AC   ## STORE          , ireg_00045     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00045     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00044     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00044     , -              , -              
##  3AC   ## BR             , ENDIF_00002    , -              , -              
j       ENDIF_00002
##  3AC   ## IF_TRUE_00002:

IF_TRUE_00002:
##  3AC   ## CALL_PROC      , print_int      , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LOAD           , ireg_00047     , ($FP)          , 4              
lw           $t3,    ($fp)
##  3AC   ## STORE          , ireg_00047     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00047     , -              , -              
##  3AC   ## JAL            , print_int      , -              , -              
jal     print_int
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00046     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00046     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LI             , ireg_00049     , 58             , -              
li           $t3,       58
##  3AC   ## STORE          , ireg_00049     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00049     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00048     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00048     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LI             , ireg_00051     , 32             , -              
li           $t3,       32
##  3AC   ## STORE          , ireg_00051     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00051     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00050     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00050     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LI             , ireg_00053     , 102            , -              
li           $t3,      102
##  3AC   ## STORE          , ireg_00053     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00053     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00052     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00052     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LI             , ireg_00055     , 98             , -              
li           $t3,       98
##  3AC   ## STORE          , ireg_00055     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00055     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00054     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00054     , -              , -              
##  3AC   ## CALL_PROC      , print_char     , 0              , -              
CALLER_FUNCTION_PROLOGUE()
##  3AC   ## LI             , ireg_00057     , 10             , -              
li           $t3,       10
##  3AC   ## STORE          , ireg_00057     , ($SP)          , 4              
sw           $t3,    ($sp)
##  3AC   ## SUB            , $SP            , $SP            , 4              
sub          $sp,      $sp,        4
##  3AC   ## KICK           , ireg_00057     , -              , -              
##  3AC   ## JAL            , print_char     , -              , -              
jal     print_char
##  3AC   ## END_PROC       , 0              , -              , -              
CALLER_FUNCTION_EPILOGUE()
##  3AC   ## ADD            , ireg_00056     , $RV            , $ZERO          
add          $t3,      $v0,    $zero
##  3AC   ## KICK           , ireg_00056     , -              , -              
##  3AC   ## ENDIF_00002:

ENDIF_00002:
##  3AC   ## KICK           , ireg_00035     , -              , -              
##  3AC   ## ENDIF_00000:

ENDIF_00000:
##  3AC   ## KICK           , ireg_00011     , -              , -              
##  3AC   ## LA             , ireg_00058     , ($FP)          , -              
la           $t0,    ($fp)
##  3AC   ## LOAD           , ireg_00059     , (ireg_00058)   , 4              
lw           $t1,    ($t0)
##  3AC   ## ADD            , ireg_00060     , ireg_00059     , $ZERO          
add          $t3,      $t1,    $zero
##  3AC   ## ADDIU          , ireg_00059     , ireg_00059     , 1              
addiu        $t1,      $t1,        1
##  3AC   ## STORE          , ireg_00059     , (ireg_00058)   , 4              
sw           $t1,    ($t0)
##  3AC   ## KICK           , ireg_00058     , -              , -              
##  3AC   ## KICK           , ireg_00059     , -              , -              
##  3AC   ## KICK           , ireg_00060     , -              , -              
##  3AC   ## BR             , LOOP_CONDITION_00000, -              , -              
j       LOOP_CONDITION_00000
##  3AC   ## LOOP_EXIT_00000:

LOOP_EXIT_00000:
## SOURCE ## 37:                    }
## SOURCE ## 38:                 }
## SOURCE ## 39: 
## SOURCE ## 40:                 return 0;
##  3AC   ## LI             , ireg_00061     , 0              , -              
li           $t3,        0
##  3AC   ## RETURN         , ireg_00061     , -              , -              
add          $v0,      $t3,    $zero
CALLEE_FUNCTION_EPILOGUE()
##  3AC   ## ENDPROC        , -              , -              , -              
CALLEE_FUNCTION_EPILOGUE()
##  3AC   ## PROG_END:
## SOURCE ## 41:             }
## SOURCE ## 42:             

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