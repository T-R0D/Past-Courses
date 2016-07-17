////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      
//  Created By: 
//  Reviewed By:
//  Course:     
//
//  Summary:   
// 
//  Last Modified: 
//
////////////////////////////////////////////////////////////////////////////////

//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// headers/namespaces
#include <cstdlib>
#include <iostream>
using namespace std;

#include "ListLinked.cpp"


//============================================================================//
//= Global Constants =========================================================//
//============================================================================//

struct GardenSummary {
 public:
  int num_patches;
  List<int> patch_sizes; 
};

//============================================================================//
//= Function Prototypes ======================================================//
//============================================================================//

void ReadInGarden( int* the_garden, const int row_size, 
                   const int column_size );

int TagPatch( int* the_garden, const int current_index, const int tag,
              int row_size, int column_size );

void DoOutput( const int num_gardens, 
               List<GardenSummary>& garden_summaries );

//============================================================================//
//= Main Function ============================================================//
//============================================================================//

int main() {
  // variables
  int row_size = 0;
  int column_size = 0;
  char char_catcher = ' ';

  int* the_garden = NULL;
  int garden_index = 0;
  int num_patches = 0;

  int num_gardens = 0;


  int current_tag = 1;
  int num_tagged = 0;

  GardenSummary* summary = NULL;
  List<GardenSummary> garden_summaries;

  bool keep_processing = true;

  // as long as there are gardens, process them
  while( keep_processing ) {
    // read garden size
    cin >> column_size;
    cin >> row_size;

      // if the garden is 0x0, stop executing
      if( (row_size == 0) || (column_size == 0) ) {
        keep_processing = false;  // possibly redundant
        break;
      }

    // create new garden
      // get memory
      the_garden = new int [row_size * column_size];

      // increment garden count
      ++ num_gardens;

      // get new summary object
      summary = new GardenSummary;
        summary->num_patches = 0;
        summary->patch_sizes.clear();

      // read in a garden
      ReadInGarden( the_garden, row_size, column_size );

    // process the garden by iteration
    for( garden_index = 0; garden_index < (row_size * column_size);
         ++ garden_index ) {
      // tag pumpkin patches, get patch counts
      num_tagged = TagPatch(the_garden, garden_index, current_tag, row_size,
                           column_size);

      // case: a patch was tagged
      if( num_tagged > 0 ) {
        // add the count to the list
        summary->num_patches ++;
        summary->patch_sizes.insert(num_tagged);

        // increment the tag
        ++ current_tag;
      }
    }

    // save the data that was gathered
    garden_summaries.insert( *summary );

    // return memory
    delete [] the_garden;
      the_garden = NULL;
    delete summary;
      summary = NULL;
  }

  // output results for each garden
  DoOutput( num_gardens, garden_summaries );




  // end program
  return 0;
}


//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//

void ReadInGarden( int* the_garden, const int row_size, 
                   const int column_size ) {
  // variables
  int row_index = 0;
  int column_index = 0;
  int garden_index = 0;
  char plot_tag = 'a';


  // iterate through the input
  // start with input rows
  for( row_index = 0;
       row_index < column_size && garden_index < (row_size * column_size);
       ++ row_index ) {

    // iterate across row members
    for( column_index = 0; column_index < row_size; 
         ++ column_index, ++ garden_index) {
      // get each character
      cin >> plot_tag;

      // case:pumpkin
      if( plot_tag == 'p' ) {
        // store -1 in garden
        the_garden[garden_index] = -1;
      }
      // case: not pumpkin
      else {
        // store 0 in garden
        the_garden[garden_index] = 0;
      }
    }
  }

  // no return - void
}

int TagPatch( int* the_garden, const int current_index, const int tag,
              int row_size, int column_size ) {
  // variables
  int tag_count = 0;
  int garden_size =  (row_size * column_size);

  // check current index to see if it is an untagged pumpkin
  if( the_garden[current_index] < 0 ) {
    // tag plot with specified number
    the_garden[current_index] = tag;
    ++ tag_count;

    // if necessary, tag adjacent plots with same number
    // checking may seem odd since a 2-D object has been abtracted to a 
    // 1-D structure

      // check north, there is no north for first row_size elements
      if( (current_index - row_size) >= 0 ) {
        // check the plot
        tag_count += TagPatch( the_garden, current_index - row_size,
                               tag, row_size, column_size );
      }

      // check south, there is no south for last row_size elements
      if( (current_index + row_size) < garden_size ) {
        // check the plot
        tag_count += TagPatch( the_garden, current_index + row_size,
                               tag, row_size, column_size );
      }

      // check east, there is no east for elements at ends of rows
      if( ((current_index + 1) < garden_size)
          && ((current_index % row_size) != (row_size - 1)) ) {
        // check the plot
        tag_count += TagPatch( the_garden, current_index + 1, tag, row_size,
                               column_size );
      }

      // check west, there is no west for first elements in rows
      if( ((current_index - 1) >= 0)
          && ((current_index % row_size) != 0) ) {
        // check the plot
        tag_count += TagPatch( the_garden, current_index - 1, tag, row_size,
                               column_size );
      }
  }

  // BASE CASE: plot is not a pumpkin or is already tagged
    // do nothing

  // return size of the patch
  return tag_count;
}


void DoOutput( const int num_gardens, 
               List<GardenSummary>& garden_summaries ) {
  // variables
  int garden_num = 1;
  int num_patches = 0;
  int patch_index = 0;
  List<int> patch_list;

  // for every garden given, output ordered patch sizes
  for( garden_num = 1, garden_summaries.gotoBeginning(); 
       garden_num <= num_gardens; ++ garden_num ) {
    // get a pointer for the patch size list and the number of patches
    patch_list = garden_summaries.getCursor().patch_sizes;

    // output the garden number
    cout << "Garden # " << garden_num << ": ";

    // output its number of patches
    num_patches = garden_summaries.getCursor().num_patches;
    cout << num_patches << " patches, ";

    // output each of its patch sizes
      // sort before outputting
      patch_list.BubbleSortL2H();

      // do the output
      cout << "sizes: ";
      for( patch_list.gotoBeginning(), patch_index = 0; patch_index < num_patches;
           ++ patch_index) {
        // output a patch size
        cout << patch_list.getCursor() << " ";

        // move on to the next patch
        patch_list.gotoNext();
      }

    // output one endline
    cout << endl;

    // move to next summary
    garden_summaries.gotoNext();
  }

  // no return - void
}
