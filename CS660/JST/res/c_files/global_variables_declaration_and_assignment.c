
            const int GLOBAL_CONST = 4;
            int GLOBAL_VAR = 2;

            int main() {

                // print the values that will be in the variables
                print_int(GLOBAL_CONST);    // expect to see 4
                print_int(GLOBAL_VAR);      // expect to see 2

                // perform the assignment
                GLOBAL_VAR = GLOBAL_CONST;
                print_int(GLOBAL_VAR);      // expect to see 4

                return 0;
            }


            