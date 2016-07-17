 /* DEFINITIONS */
%{
    /* called anytime YACC finds an error */
    void yyerror(const char *str) {
        fprintf(stderr,"error: %s\n",str);
    }
     
    /* this can define how to keep reading from additional files */
    int yywrap() {
        return 1;
    }

    /* this main function kicks off the YACCing */ 
    main() {
        yyparse();
    } 
%}

%token NUMBER L_PAREN R_PAREN PLUS MINUS TIMES DIVIDE
       POW QUIT

%%
 /* RULES */

%start input :
    /* empty */
    |
    input line
    ;

line :
    NEWLINE
    |
    expression NEWLINE
    ;

expression :
    NUMBER
    |
    expression PLUS expression
    |
    expression MINUS expression
    |
    expression TIMES expression
    |
    expression DIVIDE expression
    |
    MINUS expression %prec NEG
    |
    expression POW expression
    |
    L_PAREN expression R_PAREN
    ;

%%