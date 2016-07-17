const int n_items = 5;

void
bubble_sort(int* items, int num_items);

int
main(int argc, char** argv) {
  int* items = (int*) malloc(n_items * sizeof(int));
  int i;
  for (i = 0; i < n_items; i++) {
    items[i] = n_items - i;
  }

  bubble_sort(items, n_items);

  return 0;
}


void
bubble_sort(int* items, int num_items) {
  // This is the 'textbook' implementation and not the awesome optimized one
  int i = 0;
  int j = 0;
  int temp;

  for (i = 0; i < n_items; i++) {
    for (j = i + 1; j < n_items; j++) {
      if (items[i] > items[j]) {
        temp = items[i];
        items[i] = items[j];
        items[j] = temp;
      }
    }
  }
}
