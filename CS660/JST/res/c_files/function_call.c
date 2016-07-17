
            int foo( int a, char b);

            int main()
            {
                int i = foo(1,'a');
                print_int(i);        // expect to see 123
                return 0;
            }

            int foo( int a, char b)
            {
                print_int(a);        // expect to see 1
                return 123;
            }
            