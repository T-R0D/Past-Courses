 /* Note that comments usually need to be proceeded by a blank (in fact,
    all C code needs to be preceeded by whitespace).
  */

%{
 /* C headers and such go between the %{ %} */
#include <stdio.h>
 /* this gives us the token names as defined by our grammar in the YACC file */
#include "y.tab.h"
%}

 /* In this section we can define C variables and shorthands for Regular
    expressions
  */
 int x;

D [0-9]

%%
 /* Lex Regular expressions and code indicating what to do when encountered
    go in the following section
  */

 /* Multi-line functions can be defined by placing them in the {}s.
    We are basically writing a function body for a function that passes
    in char* yytext and returns void* yylval.
  */
D+ {
    yylval = atoi(yytext);
    return NUMBER;
}
heat                    {return TOKHEAT;}
on|off {
    yylval = !strcmp(yytext, "on");
    return STATE;
}
target                  {return TOKTARGET;}
temperature             {return TOKTEMPERATURE ;}
\n                      {/* ignore end of line */ ;}
[ \t]+                  {/* ignore whitespace */ ;}
%%
