
            int main() {
                int local_variable;

                // print the garbage that will be in the variable
                print_int(local_variable);      // will most likely see 0 but could be different since its garbage

                // perform an assignment and print to show that the
                // value was assigned
                local_variable = 123;           // expect to see 123
                print_int(local_variable); print_char('\n');

                // assign another value to show that it can be overwritten
                local_variable = 126;           // expect to see 126
                print_int(local_variable);

                return 0;
            }
            