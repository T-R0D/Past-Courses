
            int main() {

                int i = 1;
                int j = 1;

                j = i++;
                print_int(j);   // expect to see 1
                print_int(i);   // expect to see 2

                j = ++i;
                print_int(j);   // expect to see 3
                print_int(i);   // expect to see 3


                return 0;
            }
            