long int iter_Fact( int number);

int main() {
  int number;
  long int fact;

  printf( "Enter number to get factorial of: ");
  scanf( "%d", &number );

  fact = iter_Fact(number);

  printf( "Factorial of %d is:  %ld \n", number, fact);

  return 0;
}

long int iter_Fact( int number) {
  int i;
  long int fact = 1;

  if( i < 0){
    return 1;
  }

  for( i = number; i > 0; i --) {
    fact = fact*i;
  }
  return fact;
}
