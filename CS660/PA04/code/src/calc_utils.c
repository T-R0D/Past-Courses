#include "calc_utils.h"

#include <errno.h>
#include <limits.h>

int extract_int(int* error, char* yytext, const int yyleng) {
  char* end = yytext + yyleng;
  *error = 0;

  int number_value = strtol(yytext, &end, 10);

  if (errno == ERANGE || number_value > INT_MAX) {
    // the documentation states that the returned value should
    // be LONG_[MIN|MAX] if there is a conversion failure, but I have found
    // that this is not the case. It seems that checking errno is the most
    // reliable method.
    //
    // also, since a long can differ in size from an int, we check for that too
    *error = 1;
  }

  return number_value;
}

int safe_add(int* error, int x, int y) {
  int sum = 0;
  int bits_in_int = (sizeof(int) * 8) - 1;
  *error = 0;

  if (number_of_bits(x) < bits_in_int &&
      number_of_bits(y) < bits_in_int) {
    sum = x + y;
  } else {
    *error = 1;
  }

  return sum;
}

int safe_multiply(int* error, int x, int y) {
  int product = 0;
  int bits_in_int = (sizeof(int) * 8) - 1;
  *error = 0;

  if (number_of_bits(x) + number_of_bits(y) <= bits_in_int) {
    product = x * y;
  } else {
    *error = 1;
  }

  return product;
}


int
number_of_bits(int n) {
  int n_bits = 0;
  
  if (n < 0) {
    n *= -1;
  }

  while (n > 0) {
    n_bits += 1;
    n >>= 1;
  }

  return n_bits;
}
