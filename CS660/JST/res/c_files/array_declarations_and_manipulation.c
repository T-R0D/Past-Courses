            int main() {

                int i[3];
                int j[2][2];
                int k[2][2][2][2][2];
                int s;

                // 1-D manipulation
                i[0] = 2;
                i[2] = i[0];
                for( s = 0; s < 3; s++) {
                    print_int(i[s]);    // expect to see 2, 0, 2
                }

                // 2-D manipulation
                j[0][0] = 20;
                print_int(j[0][0]); // expect to see 20
                j[1][1] = j[0][0];
                print_int(j[1][1]); // expect to see 20


                // 5-D manipulation
                k[1][0][1][0][1] = 45;
                print_int(k[1][0][1][0][1]);    // expect to see 45
                k[0][0][0][0][1] = k[1][0][1][0][1];
                print_int(k[0][0][0][0][1]);    // expect to see 45


                return 0;
            }