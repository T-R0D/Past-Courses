
            int main() {
                /**
                 * The following test is an interesting composition of loops.
                 * First the _do-while_ runs through all of its iterations.
                 * Then the _while_ loop runs, but with each of its iterations, a "do" is
                 * forced, so we see a _do-while_ iteration along with each _while_ iteration.
                 * Finally, the _for_ is allowed to progress without additional _while_ iterations.
                 * This is further discussed in the comments on each line
                 */

                int i = 0;
                int j = 10;
                int k = 15;

                // test for loop
                for( i = 0; i <= 5; i ++ ) {

                    // test while loops
                    while( j <= 15 ) {

                        // test do while loops
                        do{
                            // expect to see 15-20, then 21-25 interspersed with numbers from js
                            print_char('k'); print_char(':'); print_char(' '); print_int(k);
                            print_char('\n');
                            k++;
                        } while( k <= 20 );

                        // expect to see 10-15 interspersed with the numbers from the ks do
                        print_char('j'); print_char(':'); print_char(' '); print_int(j);
                        print_char('\n');

                        j++;
                    }

                    // expect to see 0-5
                    print_char('i'); print_char(':'); print_char(' '); print_int(i);
                    print_char('\n');
                }

                return 0;
            }
            