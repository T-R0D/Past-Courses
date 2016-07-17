
            int main() {

                int i = 0;
                int j = 0;
                int k = 0;

                i = i + 10; print_int(i); print_char('\n'); // prints 10
                i = i - 2;  print_int(i); print_char('\n'); // prints 8
                i = i * 2;  print_int(i); print_char('\n'); // prints 16
                i = i / 4;  print_int(i); print_char('\n'); // prints 4
                i = i % 3;  print_int(i); print_char('\n'); // prints 1
                print_char('\n');

                j = i++;
                // prints 1 and 2
                print_int(j); print_char(' '); print_int(i); print_char('\n');

                j = ++i;
                // prints 3 and 3
                print_int(j); print_char(' '); print_int(i); print_char('\n');
                print_char('\n');

                j = i--;
                // prints 3 and 2
                print_int(j); print_char(' '); print_int(i); print_char('\n');

                j = --i;
                // prints 1 and 1
                print_int(j); print_char(' '); print_int(i); print_char('\n');
                print_char('\n');

                j += i;
                // prints 2
                print_int(j); print_char('\n');

                j -= i;
                // prints 1
                print_int(j); print_char('\n');
                print_char('\n');

                k = i = j;
                // prints 1 1 1
                print_int(k); print_char(' '); print_int(i); print_char(' '); print_int(j);
                print_char('\n');
                print_char('\n');

                i = i && 0;
                print_int(i); print_char('\n'); // prints 0

                i = 1 && 1;
                print_int(i); print_char('\n'); // prints 1
                print_char('\n');

                j = i || 5;
                print_int(j); print_char('\n'); // prints 1

                j = 0 || 0;
                print_int(j); print_char('\n'); // prints 0

                return 0;
            }
            