
            int main() {
                int local_variable = 1;
                int other_variable = 9;
                int third_variable = 5;

                // print the values that will be in the variables
                print_int(local_variable);  // expect to see 1
                print_int(other_variable);  // expect to see 9
                print_int(third_variable);  // expect to see 5

                // perform the addition
                print_int(local_variable + other_variable);         // expect to see 10
                third_variable = local_variable + other_variable;
                print_int(third_variable);                          // expect to see 10, again

            }
            