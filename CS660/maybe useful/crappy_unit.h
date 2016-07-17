#ifndef _CRAPPY_UNIT_H_
#define _CRAPPY_UNIT_H_ 1

/**
 * CraPPy-Unit
 * 
 * A single file unit-test framework  for C/CPP that's just small enough to
 * not be very good, but just complex enough to be barely annoying.
 *
 * CraPPy-Unit is designed to be very easy to use and makes use of function
 * pointers so that all you have to do is write a plain old function to run
 * it as a unit test.
 *
 * To write a test, declare a function that takes no parameters and returns an
 * int:
 *
 *  int test_some_unit_test(); // prefixing with "test" is purely convention,
 *                             // nothing will happen if you don't do it.
 *
 * then do whatever you need to in the function, and return a CrappyOutcome
 * as appropriate (e.g. CRAPPY_PASS, CRAPPY_FAIL).
 *
 * Run all of your tests in a "main file" somewhere. Initialize a CrappyResults
 * struct, and run your tests simply by passing the name of your unit test
 * funtction. A very simple "main file" would be:
 *
 * #include <stdio.h>
 *
 * #include "crappy_unit.h"
 *
 * int test_always_passes() {return CRAPPY_PASS;}
 *
 * int main() {
 *  CrappyResults* crappy_results =
 *    (CrappyResults*) malloc(sizeof(CrapResults));
 *  crappy_init(crappy_results);
 *
 *  crappy_test(crappy_results, "Always Passes", test_always_passes);
 *
 *  crappy_summary(crappy_results)
 *
 *  crappy_destroy(crappy_results);
 *  return 0;
 * }
 *
 * Note that if you want specific error messages, currently, you need to print
 * them. It is recommended that you start a new line, prepend with some
 * white space, add some white space, and then start another new line
 * followed by some more whitespace:
 *    "\n    Specific failure message.\n    "
 */

#include <string.h>


const int MESSAGE_LEN = 256;
const int MAX_TEST_NAME_WIDTH = 45;


struct CrappyResults_t {
  int tests_passed;
  int tests_failed;
  char* report;
  int report_size;
};


typedef struct CrappyResults_t CrappyResults;
typedef int (*TestFunction)();


enum CrappyOutcomes {
  CRAPPY_PASS,
  CRAPPY_FAIL
};


static void
__append_to_report(struct CrappyResults_t* self, const char* message) {

  if (strlen(message) + strlen(self->report) + 1 > self->report_size) {
    char* temp = self->report;
    self->report_size *= 2;
    self->report = (char*) malloc(self->report_size * sizeof(char));
    strcpy(self->report, temp);
  }

  strcat(self->report, message);
}

void
crappy_init(struct CrappyResults_t* self) {
  self->tests_passed = 0;
  self->tests_failed = 0;
  self->report_size = MESSAGE_LEN;
  self->report = (char*) malloc(self->report_size * sizeof(char));
}

void
crappy_destroy(struct CrappyResults_t* self) {
  free(self->report);
  self->report = NULL;
}


void
crappy_test(struct CrappyResults_t* self, const char* test_name,
            TestFunction test_function) {

  printf("%s...", test_name);

  int result = test_function();

  char result_message[6];
  if (result == CRAPPY_PASS) {
    self->tests_passed += 1;
    strcpy(result_message, "pass ");
  } else {
    self->tests_failed += 1;
    strcpy(result_message, "FAIL!");
  }
  printf(
    "%*s\n", (int) (MAX_TEST_NAME_WIDTH - strlen(test_name)), result_message);
}

void
crappy_summary(struct CrappyResults_t* self) {
  int tests_run = self->tests_passed + self->tests_failed;

  printf(
    "==============================\n"
    "||    CRAPPY TEST RESULTS   ||\n"
    "==============================\n"
    "\n"
    "Tests run:     %d\n"
    "Tests passed:  %d\n"
    "Tests FAILed:  %d\n"
    "%% passed:      %.2f%%\n"
    "\n"
    "%s\n",
    tests_run,
    self->tests_passed,
    self->tests_failed,
    (float) self->tests_passed * 100.0 / (float) tests_run,
    self->tests_failed == 0
      ? "I guess you didn't do _that_ CraPPy..." :
        "You have some things you need to fix."
    );
}

#endif
