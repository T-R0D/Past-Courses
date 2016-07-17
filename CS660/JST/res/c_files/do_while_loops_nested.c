
            int main() {

                int l = 20;
                int m = 25;

                // test do while loops
                do{
                  print_int(l);     // expect to see 20, then the m's, then 21-24 interspersed with the m's do
                  l++;

                  do{
                    print_int(m);   // expect to see 25-29 then 30-33 interspersed with numbers from l's do
                    m++;

                  }while ( m < 30 );

                }while (l < 25);

                return 0;
            }
            