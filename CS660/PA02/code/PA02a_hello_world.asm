#
# MIPS Hello World
#
# Author: Terence Henriod
#

# The data must be declared in its own section
.data
hello_string: .ascii "Hello World!\n\0"
	
# The actual program instructions need their own section
.text
main:
    li $v0, 4  # load the syscall for printing (4) into the first
	       # syscall argument register
    la $a0, hello_string  # load the address (pointer) to the
		          # string to be printed into the first
		          # syscall argument register
    syscall # execute the system call as we have set it up
    
    li $v0, 10  # prepare to use the program exit syscall (10)
    syscall  # execute the program termination