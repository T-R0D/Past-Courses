# some helpful macros
.macro print_int(%value)
	li $v0, 1
	li $a0, %value
	syscall
.end_macro

# str_addr is the base address of a null terminated ascii string
.macro print_str(%str_addr)
	li $v0, 4
	la $a0, %str_addr
	syscall
.end_macro

.macro LAND(%lhs, %rhs)
	beqz %lhs, FALSE
	beqz %rhs, FALSE
	li $t7, 1
	j END
	FALSE:
	li $t7, 0
	END:
.end_macro

.macro LOR(%lhs, %rhs)
	bnez %lhs, TRUE
	bnez %rhs, TRUE
	li $t7, 0
	j END
	TRUE:
	li $t7, 1
	END:
.end_macro


.data
	# proper declarations of globals
	g: .word 5          # int g = 5
	fp1: .double 4.5    # double fp1 = 4.5
	my_array: .space 5  # char my_array[5];
	
	# since we can't load floating points with li, they might need to be declared this way
	d1: .double 1.1
	d2: .double 2.2

.text
	# load some doubles into the coprocessor 1 registers and add them
	# double word types are referenced by even number registers, but
	# use the register below them also, i.e. $f2 and $f1
	ldc1 $f2, d1
	ldc1 $f4, d2
	add.d  $f6, $f2, $f4

	li.s $f0, 3.2

	li $t1, 0xFF
	sw $t1, 0x10010001

	lwc1 $f0, fp1

	cvt.w.s $f1, $f0
	mfc1 $t0, $f1
