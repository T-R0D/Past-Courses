int
main(int argc, char** argv) {
  int x = 5;

  int result = iterative_factorial(x);

  return 0;
}

int
iterative_factorial(int x) {
  int result = 1;
  while (x > 0) {
    result *= x;
    x--;
  }

  return result;
}
