
            int factorial(int x);

            int main() {
              int x = 5;
              int result = factorial(x);
              print_int(x);       // expect to see 5
              print_int(result);  // expect to see 120

              return 0;
            }

            int factorial(int x) {
              if (x > 1) {
                return x*factorial(x - 1);
              } else {
                return 1;
              }
            }
            