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
    yyerror("The number (%s) is not properly formatted.", yytext);
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

. {/* unrecognized characters are errors */ return ERROR;}

%%
 /* USER CODE */

  int extract_int(int* error, char* yytext, const int yyleng) {
    char* end = yytext + yyleng;
    *error = 0;

    int number_value = strtol(yytext, &end, 10);

    if (errno == ERANGE) {
      // the documentation states that the returned value should
      // be LONG_[MIN|MAX] if there is a conversion failure, but I have found
      // that this is not the case. It seems that checking errno is the most
      // reliable method.
      *error = 1;
    }

    return number_value;
  }
