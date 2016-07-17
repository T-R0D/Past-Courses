#ifndef CALC_UTILS_H
#define CALC_UTILS_H value

int
extract_int(int* error, char* yytext, const int yyleng);

int
safe_add(int* error, int x, int y);

int
safe_multiply(int* error, int x, int y);

#endif
