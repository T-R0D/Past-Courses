CC	= cc
LEX	= flex
YACC = bison
 
# WD = Working Directory
# Only works if project root dir is the working directory
WD = $(shell pwd)
SRC = $(WD)/src
OUT = $(WD)/out
BIN = $(WD)/bin

LIBS = -L/usr/local/lib -I$(OUT) -lfl

OBJS = $(OUT)/calc.lex.yy.o $(OUT)/calc.tab.o
TESTOBJS = driver.o $(OUT)/lex.yy.o


calc: calc.lex.yy.o
	$(CC) -o $(BIN)/$@ $(OBJS) $(LIBS)

calc.lex.yy.o: calc.lex.yy.c
	$(CC) -o $(OUT)/$@ -c $(OUT)/calc.lex.yy.c -I$(OUT) $(LIBS)

calc.lex.yy.c: $(SRC)/calc.lex
	$(LEX) -o $(OUT)/$@ $<

clean:
	-rm -f $(OUT)/calc.lex.yy.* $(BIN)/calc *~

.SUFFIXES: .c .o

.c.o:
	$(CC) -c $< -o $@
