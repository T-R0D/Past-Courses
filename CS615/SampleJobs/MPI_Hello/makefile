CC = /usr/lib64/openmpi/bin/mpic++
CFLAGS = -Wall -g
LIBS = 
EXEC = hello
HDRS = ${wildcard *.h}
SRCS = ${wildcard *.cpp}
OBJS = ${SRCS:.cpp=.o}

${EXEC}: ${OBJS}
	${CC} ${LIBS} -o ${EXEC} ${OBJS}

%.o: %.cpp
	${CC} ${LIBS} ${CFLAGS} -c $*.cpp

${OBJS}: ${HDRS}

clean:
	rm -f ${OBJS} ${EXEC}
