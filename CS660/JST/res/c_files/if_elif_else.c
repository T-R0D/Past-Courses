
            int main() {
                int i = 0;

                // FizzBuzz
                for( i = 1; i <= 30; i++) {
                   //FizzBuzz
                   if( i % 3 == 0){

                        //FizzBuzz
                        if( i % 5 == 0 ){
                            // expect to see this at 15 and 30
                            print_int(i); print_char(':'); print_char(' ');
                            print_char('f'); print_char('b');
                            print_char('\n');
                        }

                        //Fizz
                        else {
                           // expect to see this at 3,6,9,12,18,21,24,27
                           print_int(i); print_char(':'); print_char(' ');
                           print_char('f');
                           print_char('\n');
                        }

                   }
                   // Buzz
                   else if( i % 5 == 0) {
                       // expect to see this at 5,10,15,20,25
                       print_int(i); print_char(':'); print_char(' ');
                       print_char('b');
                       print_char('\n');
                   }
                   // Number
                   else {
                       // expect to see all other numbers except those mentioned above
                       print_int(i); print_char('\n');
                   }
                }

                return 0;
            }
            