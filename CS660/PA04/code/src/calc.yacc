 /* DECLARATIONS */
%{
  #include <stdio.h>
  #include "calc_utils.h"
%}

%start line

%token INTEGER OPEN CLOSE PLUS MINUS MULT DIV SEMI ERROR

 // precedences
%left PLUS MINUS
%left MULT DIV

%%
 /* RULES */
line:
  /* empty */
  |
  line expression SEMI {
    printf("%d\n", $2);
  }
  ;

expression:
  expression PLUS term {
    int error = 0;
    int result = safe_add(&error, $1, $3);

    if (error != 0) {
      yyerror("Addition overflow.");
    } else {
      $$ = result;
    }
  }
  |
  expression MINUS term {
    $$ = $1 - $3;
  }
  |
  term {
    $$ = $1;
  }
  ;

term:
  term MULT factor {
    int error = 0;
    int result = safe_multiply(&error, $1, $3);

    if (error != 0) {
      yyerror("Multiplication overflow.");
    } else {
      $$ = result;
    }
  }
  |
  term DIV factor {
    if ($3 == 0) {
      yyerror("Division by zero is undefined.");
    } else {
      $$ = $1 / $3;
    }
  }
  |
  factor {
    $$ = $1;
  }
  ;

factor:
  OPEN expression CLOSE {
    $$ = $2;
  }
  |
  INTEGER {
    $$ = $1;
  }
  ;


%%
 /* USER CODE */
int main(int argc, char** argv) {
  return yyparse();
}

yyerror(char* s) {
  fprintf(stdout, "%s\n",s);
  exit(1);
}

yywrap() {
  // return 1 for funished with reading input
  return 1;
}
