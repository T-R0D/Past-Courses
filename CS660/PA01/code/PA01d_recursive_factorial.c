int
main(int argc, char** argv) {
  int x = 5;

  int result = recursive_factorial(x);

  return 0;
}

int
recursive_factorial(int x) {
  int result;

  if (x < 0) {
    result = -1;
  } else if (x <= 1) {
    result = 1;
  } else {
    result = x * recursive_factorial(x - 1);
  }

  return result;
}
