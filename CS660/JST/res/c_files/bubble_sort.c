
            const int N_ITEMS = 5;

            int main() {
              int i, j;
              int temp;
            // int things[N_ITEMS] = {5, 1, 4, 3, 2};

              int things[N_ITEMS];
              things[0] = 5;
              things[1] = 1;
              things[2] = 4;
              things[3] = 3;
              things[4] = 2;

              print_int(things[0]);  // expect to see 5
              print_int(things[1]);  // expect to see 1
              print_int(things[2]);  // expect to see 4
              print_int(things[3]);  // expect to see 3
              print_int(things[4]);  // expect to see 2

              for (i = 0; i < N_ITEMS; i++) {
                for (j = i; j < N_ITEMS; j++) {
                  if (things[i] < things[j]) {
                    temp = things[i];
                    things[i] = things[j];
                    things[j] = temp;
                  }
                }
              }

              print_int(things[0]);  // expect to see 5
              print_int(things[1]);  // expect to see 4
              print_int(things[2]);  // expect to see 3
              print_int(things[3]);  // expect to see 2
              print_int(things[4]);  // expect to see 1

              return 0;
            }
            