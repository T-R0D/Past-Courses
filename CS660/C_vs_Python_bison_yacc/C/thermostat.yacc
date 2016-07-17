%{
#include <stdio.h>
#include <string.h>
 
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

/* Define the tokens to be used. YACCing with -d will put these in
   y.tab.h.
 */
%token NUMBER TOKHEAT STATE TOKTARGET TOKTEMPERATURE

%%
commands: /* empty */
        | commands command
        ;

command:
        heat_switch
        |
        target_set
        ;

heat_switch:
        TOKHEAT STATE
        {
            if ($2 == 0) {
                printf("\tHeat turned OFF\n");
            } else {
                printf("\tHeat turned ON\n");
            }
        }
        ;

target_set:
        TOKTARGET TOKTEMPERATURE NUMBER
        {
                printf("\tTemperature set\n");
        }
        ;
