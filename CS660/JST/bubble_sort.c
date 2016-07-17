// function prototypes
void print( int* list, int size);
void bubbleSort(int* list, int size);

// main
int main() {
   // initialize vars
   int* list = (int*)malloc(10);
   int i;

   srand(time(NULL));

   // create list
   for(i =0; i<10;i++)
      {
       list[i] = rand() % 10 + 1; 
      }
   print(list, 10);

   // bubble sort
   bubbleSort( list, 10 );

   printf( "Sorted " );
   print(list, 10);

   !!C

   // return
   return 0;

}


// fxn imp
void bubbleSort(int* list, int size){
   // initialize vars
   int i,j;
   int temp;
   int swapped;
   
   // loop through list
   for( i = 0; i < size; i++)
      {

      // swapped is false
      swapped = 0;

      // loop through list
      for( j = 0; j < size - 1; j++)
         {
         // if smaller, swap
         if( list[j+1] < list[j])
            {
            temp = list[j];
            list[j] = list[j+1];
            list[j+1] = temp;
            swapped = 1;
            }
         }
      // if swapped is false, break
      if( swapped == 0)
         {         
         break;
         }   
      }

   }

void print( int* list, int size ){
   int i;
   printf("List is: ");

   for(i =0; i < size; i++)
      {
      printf( "%d ", list[i] );
      }
   printf("\n");
}
