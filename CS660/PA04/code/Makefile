CC	= cc
LEX	= flex
YACC	= bison
LIBS	= -L/usr/local/lib -lfl 

# WD = Working Directory
# Only works if project root dir is the working directory
WD = $(shell pwd)
SRC = $(WD)/src
OUT = $(WD)/out
BIN = $(WD)/bin

LIBS = -L/usr/local/lib -I$(OUT) -I$(SRC) -lfl

OBJS = $(OUT)/calc.lex.yy.o $(OUT)/calc.tab.o $(OUT)/calc_utils.o
TESTOBJS = driver.o $(OUT)/lex.yy.o


calc: calc.tab.o calc.lex.yy.o
	$(CC) -o $(BIN)/$@ $(OBJS) $(LIBS)

test_calc: calc.tab.o calc.lex.yy.o test_calc.o
	$(CC) -o $(BIN)/$@ $(OBJS) $(OUT)/test_calc.o $(LIBS)

test_calc.o: $(SRC)/test_calc.c
	$(CC) -o $(OUT)/$@ -c $(SRC)/test_calc.c -I$(OUT)

calc.lex.yy.o: calc.lex.yy.c calc_utils.o
	$(CC) -o $(OUT)/$@ -c $(OUT)/calc.lex.yy.c -I$(OUT) -I$(SRC) -lfl

calc.lex.yy.c: $(SRC)/calc.lex
	$(LEX) -o $(OUT)/$@ $< 

calc.tab.o: calc.tab.c calc_utils.o
	$(CC) -o $(OUT)/$@ -c $(OUT)/calc.tab.c -I$(OUT) $(LIBS)

calc.tab.c: $(SRC)/calc.yacc
	$(YACC) -d  -o $(OUT)/calc.tab.c $(SRC)/calc.yacc

calc_utils.o: $(SRC)/calc_utils.c
	$(CC) -o $(OUT)/$@ -c $(SRC)/calc_utils.c

clean:
	-rm -f $(OUT)/* $(BIN)/calc *~

.SUFFIXES: .c .o

.c.o:
	$(CC) -c $< -o $@
