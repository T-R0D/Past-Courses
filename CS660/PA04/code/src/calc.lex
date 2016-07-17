 /**
  * calc.lex
  *
  * This is a lex file for defining a tokenizer for a simple calculator.
  */

 /* DEFINITIONS */
%{
 #include <stdio.h>
 #include <errno.h>
 #include <limits.h>
 #include "calc.tab.h"
#include "calc_utils.h"

 #define STROL_ERROR -1
%}

ZERO_STRING   [0]
POSITIVE_INT  [1-9][0-9]*
WHITESPACE    [ \t\r\n]+

%%
 /* RULES */

{ZERO_STRING} {
  yylval = 0;
  return INTEGER;
}

{POSITIVE_INT} {
  int error = 0;
  yylval = extract_int(&error, yytext, yyleng);

  if (error != 0) {
    yyerror("Bad number format.");
    return ERROR;
  }

  return INTEGER;
}

[;] {return SEMI;}

[(] {return OPEN;}

[)] {return CLOSE;}

[+] {return PLUS;}

[-] {return MINUS;}

[*] {return MULT;}

[/] {return DIV;}

{WHITESPACE} {/* ignore */;}

. {/* unrecognized characters are errors */
	yyerror("Unrecognized token.");
    return ERROR;}

%%
 /* USER CODE */
