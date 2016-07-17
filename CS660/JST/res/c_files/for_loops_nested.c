
            int main() {

                int n = 0;
                int p = 0;

                // test for loops
                for( n = 0; n < 5; n++) {
                    for( p = 0; p < 5; p ++ ) {
                        print_int(p);   // expect to see 0-4 then 0-4 after each increment of p
                    }
                    print_int(n); // expect to see 0-4 with the 0-4 from the p's after 0,1,2,3
                }
                return 0;
            }
            