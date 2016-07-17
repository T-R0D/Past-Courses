long int recur_Fact( int number);

int main() {
  int number;
  long int fact;

  printf( "Enter number to get factorial of: ");
  scanf( "%d", &number );

  fact = recur_Fact(number);

  printf("Factorial of %d is:  %ld\n", number, fact);

  return 0;
}

long int recur_Fact( int number) {
  // base case
  if( number <= 0)
    return 1;

  // recursive case
  else if( number > 1 ) {
    return number*recur_Fact(number-1);    
  }
}
