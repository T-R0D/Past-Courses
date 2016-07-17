
           const int ARRAY_DIM = 2;

            // hard code dimensions for simplicity
            int matrix_multiply(int C[ARRAY_DIM][ARRAY_DIM], int A[ARRAY_DIM][ARRAY_DIM], int B[ARRAY_DIM][ARRAY_DIM]);
            int print_matrix(int C[ARRAY_DIM][ARRAY_DIM]);

            int main() {
              int i, j, k;
              int sum;
              int A[ARRAY_DIM][ARRAY_DIM], B[ARRAY_DIM][ARRAY_DIM], C[ARRAY_DIM][ARRAY_DIM];

              for (i = 0; i < ARRAY_DIM; i++) {
                for (j = 0; j < ARRAY_DIM; j++) {
                  A[i][j] = B[i][j] = 2;
                }
              }

              // matrix_multiply
              matrix_multiply(C, A, B);

              // print_matrix
              print_matrix(C);

              return 0;
            }

            int matrix_multiply(int C[ARRAY_DIM][ARRAY_DIM], int A[ARRAY_DIM][ARRAY_DIM], int B[ARRAY_DIM][ARRAY_DIM])
            {
              int i, j, k, sum;

              // matrix_multiply
              for (i = 0; i < ARRAY_DIM; i++) {
                 for (j = 0; j < ARRAY_DIM; j++) {
                    sum = 0;
                    for( k=0; k < ARRAY_DIM; k++) {
                        sum = sum + A[i][k] * B[k][j];
                    }
                    C[i][j] = sum;
                 }
              }
            }

            int print_matrix( int C[ARRAY_DIM][ARRAY_DIM])
            {
              int i, j;

              // print_matrix
              for (i = 0; i < ARRAY_DIM; i++) {
                    for (j = 0; j < ARRAY_DIM; j++) {
                        print_int(C[i][j]);     // expect to see four 8's
                        print_char(' ');
                    }
                    print_char('\n');
              }
            }

            