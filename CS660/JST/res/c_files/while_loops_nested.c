
            int main() {

                int i = 0;
                int j = 10;
                int k = 15;

                // test while loops
                while( i <= 5 ) {
                    while( j <= 15 ) {
                        while( k <= 20 ) {
                            print_int(k);   // expect to see 15-20
                            k++;
                        }
                        print_int(j);   // expect to see 10-15
                        j++;
                    }
                    print_int(i);       // expect to see 0-5
                    i++;
                }

                return 0;
            }
            