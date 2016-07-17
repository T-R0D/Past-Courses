#include <stdlib.h>
#include <stdio.h>

#include "crappy_unit.h"

int test_crappy_unit_test_function() {
  if (1 == 1) {
    return CRAPPY_PASS;
  } else {
    return CRAPPY_FAIL;
  }
}

int test1() {
  return CRAPPY_PASS;
}

int test2() {
  return CRAPPY_FAIL;
}

int test3() {
  if (0 == 1) {
    return CRAPPY_PASS;
  } else {
    printf("\n    This is how a test fails gloriously!\n    ");
    return CRAPPY_FAIL;
  }
}

int main() {

  CrappyResults* results = (CrappyResults*) malloc(sizeof(CrappyResults));
  crappy_init(results);

  crappy_test(results, "test_crappy_test_function", test_crappy_unit_test_function);
  crappy_test(results, "test1", test1);
  crappy_test(results, "test3", test3);
  crappy_test(results, "test2", test2);

  crappy_summary(results);

  crappy_destroy(results);
  free(results);

  return 0;
}
