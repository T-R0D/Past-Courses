CC = /usr/lib64/openmpi/bin/mpicc
CFLAGS = -Wall -g
LIBS = 
EXEC = Sequential_Matrix_Multiply
HDRS = ${wildcard *.h}
SRCS = ${wildcard *.c}
OBJS = ${SRCS:.c=.o}

${EXEC}: ${OBJS}
	${CC} ${LIBS} -o ${EXEC} ${OBJS}

%.o: %.c
	${CC} ${LIBS} ${CFLAGS} -c $*.c

${OBJS}: ${HDRS}

clean:
	rm -f ${OBJS} ${EXEC}
