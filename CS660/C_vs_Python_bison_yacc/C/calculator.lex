 /* DEFINITIONS */
 /* We can put C definitions or shorthands for common
    REs here.
  */

%{
  #include <stdio.h>
  #include <string.h>

  #define returntoken(tok) yylval = PyString_FromString(strdup(yytext)); return (tok);
  #define YY_INPUT(buf,result,max_size) { (*py_input)(py_parser, buf, &result, max_size); }
%}

D [0-9]
    
%%
 /* RULES */

D+     { return NUMBER; }

(    { return L_PAREN; }

)    { return R_PAREN; }

+    { return PLUS; }

-    { return MINUS; }

*    { return TIMES; }

**   { return POW; }

/    { return DIVIDE; }

quit {
    printf("lex: got QUIT\n");
    yyterminate();
    returntoken(QUIT);
}
    
[ \t\v\f]             {}
[\n]        {yylineno++; returntoken(NEWLINE); }
.       { printf("unknown char %c ignored, yytext=0x%lx\n", yytext[0], yytext); /* ignore bad chars */}
    
%%
 /* FUNCTIONS */

int yywrap() {
    return 1;
}

int main(int argc, char** argv) {
    yylex();
    return 0;
}