#
# MIPS Iterative Factorial
#
# Author: Terence Henriod
#

.text
.globl main
.globl iterative_factorial

main:
    subu $sp, $sp, 4
    sw $ra, ($sp)
    subu $sp, $sp, 4
    sw $s0, ($sp)


    li $a0, 10

    jal iterative_factorial
    nop
    move $s0, $v0
 
 
    lw $s0, ($sp)
    addu $sp, $sp, 4
    lw $ra, ($sp)
    addu $sp, $sp, 4
    
    li $v0, 10 
    syscall
    
iterative_factorial:
    subu $sp, $sp, 4
    sw $ra, ($sp)
    subu $sp, $sp, 4
    sw $s0, ($sp)
    subu $sp, $sp, 4
    sw $s1, ($sp)
    
    move $s0, $a0 
    li $s1, 1
    
    blt $s0, 0, factorial_error
    nop
    factorial_loop:
        ble $s0, 1, end_factorial
        nop
        
        mul $s1, $s1, $s0
        sub $s0, $s0, 1
        
        j factorial_loop
    
    factorial_error:
        li $s1, -1
        
    end_factorial:
        move $v0, $s1
    
        lw $s1, ($sp)
        addu $sp, $sp, 4
        lw $s0, ($sp)
        addu $sp, $sp, 4
        lw $ra, ($sp)
        addu $sp, $sp, 4
    
        jr $ra
