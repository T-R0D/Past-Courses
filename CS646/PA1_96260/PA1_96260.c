/**
    @file PA1_96260.cpp

    @author 96260

    Family Tree

    @brief A program that demonstrates the use of the UNIX fork() command by
           displaying a family tree. While the use of fork() is the point of
           interest in the assignment, the part of the program that reads in
           and stores the family data in a tree may also be worth consideration.
           Note that an inefficient use of the tree structure and fork()
           command were used to allow for to simulate a parent spawning each of
           their children.

    @version Original Code 1.00 (2/6/2014) - 96260

    UNOFFICIALLY:
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
// standard headers
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL CONSTANTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#define EXPECTED_ARGC 2
#define FILE_NAME_ARG 1
#define FORK_ERROR 1
#define TRUE 1
#define FALSE 0
#define NAME_SIZE 31
#define MAX_FAMILY_SIZE 100000
#define NOT_FOUND -1
#define DEBUGGING 0

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   USER DEFINED TYPES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
@struct Family

This structure defines a node in a "first-born" family tree. The tree is a
binary tree, but, each node only points to one child (the "first-born" child)
and the next eldest sibling of the current family represented in a node.
Each family is represented by one parent's name, and if applicable, a marriage
partner's name. IMPORTANT: This struct is meant for use in an array based
tree, so it does not contain pointers to other nodes, it can be assumed that
the relationships of
$ left_child_node_ndx (or first-born) = 2 * parent_node_ndx + 1 $ and 
$ right_child_node_ndx (or sibling) = 2 * parent_node_ndx + 2 $ hold.

Such a tree would appear graphically as:

            (Adam - Eve) -> NULL
                  |
                  | Child
                  V
            (Cain - Cain's Wife) ----------> (Abel - NULL) -> NULL
                  |               Sibling          |
                  V                                V
                 ...                              NULL

@var family_head_name   A pointer to be used in the allocation of dynamic
                        memory for a parent's name. Also used to represent
                        the name of the household.
@var partner            A pointer to be used for dynamic allocation of a
                        marriage partner's name, should there be one.

*/
typedef struct Family
{
  char* family_head_name;
  char* partner;
} Family;


/**
@struct FamilyRecord

This structure acts as a node in a binary search tree in order to quickly
locate family nodes in the "first-born" tree. Because a first-born tree does
not offer efficient searching, this search tree is a companion to act as a
catalog of the house names (name for each node), for speedy location of
nodes in the other tree. Note that this tree is also used as an array-based
implementation, but as a binary search tree, so the semantic term "child"
may be used to describe the nodes that follow to the left or right of a
parent node. Once again:
$ left_child_node_ndx = 2 * parent_node_ndx + 1 $ and 
$ right_child_node_ndx = 2 * parent_node_ndx + 2 $ hold.

@var family_head_name   A pointer to dynamically allocated memory for the name
                        of a house (or node) that exists in the "first_born"
                        tree.
@var family_ndx         The index in the array at which the house named above
                        can be found in the tree mapped array.
*/
typedef struct FamilyRecord
{
  char* family_head_name;
  int family_ndx;
} FamilyRecord;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   FUNCTION PROTOTYPES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
goodCommandLineArgs

Simply ensures that the correct number of command line arguments was entered.

@param argc   The count of the arguments given to the program on the command
              line (including the program name itself).
@param argv   An array of c-strings containing the values entered on the
              command line when calling the program.

@return result   A boolean value with 0/false indicating that the arguments
                 are unacceptable for proper program operaion, and not 0/true
                 for acceptable argument values.

@pre
-# The program must be called from the command line.
-# The arguments are those containing the exact information of the command line
   arguments.

@post
-# The command line arguments will be checked and a boolean result returned
   to indicate their acceptability.

@detail @bAlgorithm
-# Only the count of the arguments is checked in this version of the program.

@code
@endcode
*/
int goodCommandLineArgs( const int argc, char** argv );


/**
loadData

Processes an input data file in order to construct a family tree of the names
contained in the file (see the Family struct definition). A pointer to the
array the tree is mapped to is returned. This array is dynamically allocated
and each element points to a "first-born" tree node. A second array based tree,
a binary search one (see FamilyRecord struct), is used to expedite the process
of locating family nodes in the first-born tree as the family tree is
constructed.

@param file_name   The name of the file to read the family data from, as a
                   c-string.

@return family_tree   A pointer used to dynamically allocate an array of
                      Family*s that are mapped as a binary tree.

@pre
-# The name passed as an argument must correspond to a valid input file for
   proper program execution.

@post
-# The family data from the file will be contained in the "first-born" tree,
   a pointer to that tree will be returned.

@detail @bAlgorithm
-# The specified file is opened.
-# If the file opening is successful, memory is allocated for the first-born
   and search catalog trees. Their elements are zeroed initially.
-# One name from the file is read in, and nodes in each tree are created
   accordingly.
-# If the name was not the only one on the line, names are read and stored
   as appropriate until that line of the file is exhausted.
-# Names/lines are read until the file contents are exhausted.
-# Once tree construction is complete, the search catalog tree is destroyed,
   the file is closed and a pointer to the first-born tree is returned.

@code
@endcode
*/
Family** loadData( char* file_name );


/**
getName

Reads one name from a file and stores it in the array pointed by parameter
char* the_name. Ignores any leading whitespace, and returns the value of the
first whitespace character after the name as a sort of flag. Will return EOF to
indicate failure.

@param input_file   A FILE* used for accessing the input file.
@param the_name     A pointer to a character array to be used as the reading
                    buffer.

@return char_read   Used to indicate the first whitespace character after the
                    name (or EOF) as a signal to guide the actions of higher
                    levels of the program.

@pre
-# Parameter input_file points to an open text file to read from.
-# Parameter the_name points to a properly allocated char array.

@post
-# The next name to be read will be stored in the buffer pointed by the_name.
   A null terminator is appended. Behavior is undefined should reading fail.
-# The whitespace charactor immediately following the name is returned. Should
   the end of the file be reached, this is EOF.

@detail @bAlgorithm
-# Whitespace characters are read and discarded until a non-whitespace character
   is encountered.
-# Characters are stored to the provided buffer until a whitespace character is
   encountered. A null terminator is appended.
-# The read whitespace character is returned as a signal. Should EOF ever
   be encountered, it is returned.

@code
@endcode
*/
int getName( FILE* input_file, char* the_name );


/**
makeFamily

A factory function for creating a family node. Initializes the members of the
node struct, and returns a pointer to the dynamically allocated memory.

@param head_of_house   A c-string containing the name of the house the node
                       represents.

@return new_family   A pointer to the dynamically allocated family node.

@pre
-# Parameter head_of_house points to a valid, null-terminated c-string.

@post
-# A Family struct node will be dynamically allocated, with the
   family_head_name pointing to a dynamically allocated c-string containing
   the house name and the partner member pointing to NULL.

@code
@endcode
*/
Family* makeFamily( char* head_of_house );


/**
addRecord

A factory function for creating a FamilyRecord for use in a binary search
tree. Initializes the members of a FamilyRecord struct and returns a pointer
to the dynamically allocated memory.

@param search_catalog    The pointer to the array that composes the search
                         catalog tree for fast location of families in the
                         "first-born" tree.
@param house_name        A pointer to a c-string that specifies the name
                         of the house a search entry in the catalog will
                         be created for.
@param family_tree_ndx   The of index where the family with the given house
                         name can be found in the "first-born" tree array.

@return new_record   A pointer to the newly allocated FamilyRecord struct.

@pre
-# Parameter search_catalog indicates a valid array to act as a tree for the
   FamilyRecord nodes.
-# Parameter house_name indicates a valid c-string.
-# Parameter family_tree_ndx indicates a valid array position for the family
   node the created search record will correspond to.

@post
-# A FamilyRecord will be created and initialized with the given values.
-# A pointer to this new node will be returned so it can be linked to the
   search tree.

@detail @Algorithm
-# Searches the binary search tree in order to find the appropriate place to
   add a node.
-# Creates a new node at the appropriate location in the search tree.

@code
@endcode
*/
FamilyRecord* addRecord( FamilyRecord** search_catalog, char* house_name,
                         int family_tree_ndx );


/**
findHousehold

Uses a search catalog in order to quickly locate a specified family in the
"first-born" tree.

@param search_catalog   The array that represents the search tree.
@param house_name       A c-string indicating what house should be found.

@return family_location   The index of the sought family in the "first-born"
                          tree.

@pre
-# Parameter search_catalog indicates a valid search tree.
-# Parameter house_name indicates a valid c-string.

@post
-# The index of the sought family in the first-born tree will be given.
-# If the name is not found in the family tree, NOT_FOUND is returned to
   indicate that the family does not exist in the search tree.

@detail @bAlgorithm
-# A typical binary tree search.

@code
@endcode
*/
int findHousehold( FamilyRecord** search_catalog, char* house_name );


/**
completeFamily

Completes a Family node by adding reading in the name of a partner from the
file, and then creating nodes for all of the children's families (if there)
are any.

@param input_file       A FILE* used to indicate what file to read from.
@param famliy_tree      A pointer to the array that comprises the "first-born"
                        family tree.
@param current_family   The index of the current family being completed.

@return read_result   Indicates the last character read to signal the state
                      of the file stream to higher level functions.

@pre
-# Parameter family_tree indicates a valid family tree array.
-# FILE* input_file points a previously opened file.
-# The first name (the head of the house) has been read from the current line
   and a node has been created to contain the partner data to be read in. 
-# Parameter current_family indicates the indiex at which the family being
   completed can be found in the "first-born" tree array.

@post
-# The marriage partner's name will be added to the currently considered
   Family struct.
-# Nodes will be created for the families of any children read in belonging
   to the current family. 

@detail @bAlgorithm
-# A marriage partner's name is read in.
-# If the end of a line or EOF is not reached, children's names will be read in
   and nodes will be added to the tree to hold these names.
-# records in the search tree are added for each child that is read in.

@code
@endcode
*/
int completeFamily( FILE* input_file, Family** family_tree,
                     int current_family );


/**
addChildrenToCatalog

Creates entries for the children of a given parent node in the search catalog
for later use. Allows children to be quickly located when the time comes to
complete their families.

@param search_catalog   A pointer to the search tree array, where the
                        FamilyRecords are kept.
@param family_tree      A pointer to the "first-born" tree array.
@param parent_ndx       The index of the parent whose children will be added
                        to the search catalog.

@pre
-# Parameters search_catalog and family_tree point to valid, tree-representing
   arrays.
-# Parameter parent_ndx indicates a parent node that has children nodes that
   need to be added to the search catalog.

@post
-# The children of the inicated parent will have FamilyRecord nodes created
   for them in the seach tree.

@detail @bAlgorithm
-# The function uses the given parent index to move to the parent's first-born
   child (a move to the left sub-tree).
-# A node is made for the child in the search tree.
-# If possible, the funciton moves to the child's next sibling (a move to the
   right sub-tree), and then the algorithm repeats at step 2.

@code
@endcode
*/
void addChildrenToCatalog( FamilyRecord** search_catalog, Family** family_tree,
                           int parent_ndx );


/**
printFamily

Recursively prints the contents of the "first-born" family tree in a manner
that represents the family as a whole. Members of similar generations will
be indented with the same number of tabs, and their children will be printed
immediately following them. Once a parent and their sub-family have been
printed, the sibling of that parent and their family will then be printed.
Ex: Adam(1111)-Eve
        Cain(1112)-Cain'sWife
            ...
        Abel(1112)

Per the assignment instructions, the program will fork() itself before printing
every child's family. The intent is that the fork() command will simulate
a parent spawning a child.

@param family_tree   A pointer to the array that constitutes the tree the
                     function will use to display a family.
@param parent_ndx    The index of the root of the current sub-tree to be
                     printed.
@param generation    The generation of the child about to be displayed.
                     Indicates the number of tabs to print.

@return procedure_success   Indicates the success of the printing. Used to
                            signal a forking failure, should one occur.

@pre
-# Parameters family_tree and parent_ndx represent a valid array/tree and node
   within the tree.
-# For proper functionality, all parents, grandparents, etc., need to be
   printed before the current call to the funciton.

@post
-# The given sub-family will be printed after the resolution of all recursive
   calls.

@detail @bAlgorithm
-# The information pertinent to the given family is printed: house_name (parent
   one), parent process id (ppid), and partner name.
-# The funciton then moves to the parent's first child (if one exists) by
   moving down the left sub-tree.
-# While there are children to print, fork() is called, and the new process
   makes a recursive call to print the child. The original process will be
   known as the "parent" and the newly created process will be known as the
   "child." Otherwise, the algorithm is done.
-# Housekeeping is performed to prevent the new process from duplicating the
   efforts of the "parent" process.
-# Once the "child" process terminates, the "parent" moves to the next child in
   the family (by following the right sub-tree), and the algorithm starts over
   at step 3.

@code
@endcode
*/
int printFamily( Family** family_tree, int parent_ndx, int generation );


/**
FunctionName

Given a pointer to a Family node, prints the relevant information for the
parents of that family, including any leading tabs needed to represent the
generation of the parent, the first parent's name, the ppid, and the
marriage partner's name (if there is one).

@param the_family   A pointer to the Family_node that contains the information
                    to be printed.
@param generation   Indicates the number of tabs that need to be printed.

@pre
-# Parameter the_family points to a valid Family node.

@post
-# The contents of the node and the ppid will be printed with the appropriate
   number of leading tabs.

@detail @bAlgorithm
-# A number of tabs representative of the family's generation is printed.
-# The house name (parent 1's) name is printed.
-# The ppid is printed in parentheses.
-# If the pointer to the partner's c-string is not NULL, that name is printed.
   Otherwise, nothing is printed.
-# An endline is printed.

@code
@endcode
*/
void printParents( Family* the_family, int generation );


/**
destroyCatalog

Recursively de-allocates the memory for all nodes in the search tree. The
original call should be followed be a call to free() to de-allocate the
array/tree if the tree will no longer be used (assuming it is dynamically
allocated).

@param catalog          The pointer to the entire array/tree.
@param current_record   The index of the current search record being considered.

@pre
-# The parameters should indicate a valid tree and index.

@post
-# The current record, and any child nodes below it, will have any associated
   memory de-allocted, leaving only a null pointer at the index of
   current_record (and all indices for the sub-tree) after the resolution of
   the current and all recursive calls.

@detail @bAlgorithm
-# A post-order tree traversal is used to de-allocate all node memory.

@code
@endcode
*/
void destroyCatalog( FamilyRecord** catalog, int current_record );


/**
genocide

Clears a family_tree, (destroys the entire family). Recursively follows the
tree and de-allocates any memory associated with Family nodes. De-allocates
the memory for the node itself.

@param family_tree      A pointer to the array/tree being cleared.
@param current_family   An index indicating the current sub-tree being cleared.

@pre
-# Parameters family_tree and current_family point to valid data.

@post
-# Upon resolution of the current and all recursive calls, all memory
   associated with the nodes of the sub-tree and the memory for the nodes
   themselves will be properly de-allocated. The pointers contained in
   the tree will be set to NULL. 

@detail @bAlgorithm
-# A post-order traversal is used to properly de-allocate the memory associated
   with the nodes.

@code
@endcode
*/
void genocide( Family** family_tree, int current_family );


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MAIN FUNCTION DEFINITION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
main

The main program driver. Accepts precisely one argument from the command line
(in addition to the program name itself), and this argument is the input
file to be used. Do note that the only error checking of the arguments is that
there are two. Read the data into a "first-born tree," and then outputs the
data in a way that satisfies the assignment requirements.

@param argc   The number of command line arguments that were passed to the
              main function.
@param argv   An array of c-strings representing the arguments passed via
              the command line.

@var program_success   Used to signal to the calling program whether or not
                       the program successfully completed.
@var family_tree       A pointer to an array that will be mapped as a tree.
                       The array will contain pointers to Family nodes.
@var pid               Used for differentiating forked processes.

@return program_success   Indicates the successful operation of the program.
                          0 for successful operation, otherwise, another
                          number that will indicate error.

@pre
-# Precisely one argument other than the program name was supplied for a valid
   input file name for the program to read from.

@post
-# Upon successful completion, a family tree representing the data in the file
   will have been read in and displayed.

@code
@endcode
*/
int main( int argc, char** argv )
{
  // variables
  int program_success = 0;
  Family** family_tree = NULL;
  pid_t pid = 0;

  // case: command-line arguments are acceptable
  if ( goodCommandLineArgs( argc, argv ) )
  {
    // read in the data
    family_tree = loadData( argv[FILE_NAME_ARG] );

    // case: the data was successfully processed
    if ( family_tree != NULL )
    {
      // output the family tree
      // fork the process to spawn a parent in keeping with the tenets of the
      // assignment
      pid = fork();

      // case: the current process is the child process
      if ( pid == 0 )
      {
        // begin printing out the family
        program_success = printFamily( family_tree, 0, 0 );
      }
      // case: the current process is the parent process
      else if ( pid > 0 )
      {
        // simply wait for the child/printing process to complete
        waitpid( pid, NULL, 0 );
      }
      // case: the forking failed
      else
      {
        // indicate such
        program_success = FORK_ERROR;
      }
    }
  }
  // case: command-line arguments are unacceptable
  else
  {
    // notify the user of the failure
    printf( "\n !!! The command line arguments were not acceptable. !!!\n\n" );
    printf( "Please see the program documentation or the help\n" );
    printf( "section of the program..." );
  }

  // deconstruct the family tree
  genocide( family_tree, 0 );
  free( family_tree );

  // return the program success status
  return program_success;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   FUNCTION IMPLEMENTATIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
  @var result   Used to indicate a boolean result of the command line
                arguments acceptability.
*/
int goodCommandLineArgs( const int argc, char** argv )
{
  // variables
  int result = FALSE;

  // case: the number of arguments is acceptable
  if ( argc == EXPECTED_ARGC )
  {
    // case: the first command-line arg is a file name
    if ( TRUE ) // file names will have no extensions
    {
      // all tests passed, the arguments are good
      result = TRUE;
    }
  }

  // return the result
  return result;
}


/**
  @var family_tree          Points to the array that houses the "first-born"
                            family tree.
  @var family_catalog       Points to the array that houses the search tree
                            for quickly locating families in the "first-born"
                            tree.
  @var input_file           A FILE* to the input file being utilized.
  @var house_name           The most recently read name from the file (by this
                            function). Used for the creation of tree nodes.
  @var read_result          Indicates the most recently read character of
                            the input file in order to signal when the
                            end of a line or end of a file has been reached.
  @var current_family_ndx   Indicates the index of the family node being
                            added/operated on.
*/
Family** loadData( char* file_name )
{
  // variables
  Family** family_tree = NULL;
  FamilyRecord** family_catalog = NULL;
  FILE* input_file = NULL;
  char* house_name;
  int read_result = EOF;
  int current_family_ndx = 0;

  // attempt to open the file
  input_file = fopen(file_name, "r");

  // case: file was opened
  if ( input_file != NULL )
  {

    // allocate dynamic memory for trees and name buffer
    family_tree = (Family**) calloc( MAX_FAMILY_SIZE, sizeof( Family* ) );
    family_catalog = (FamilyRecord**) calloc( MAX_FAMILY_SIZE,
                                              sizeof( FamilyRecord* ) );
    house_name = (char*) calloc( NAME_SIZE, sizeof( char ) );

    // seed the trees by getting the first head of household from the file
    read_result = getName( input_file, house_name );
    family_tree[0] = makeFamily( house_name );
    family_catalog[0] = addRecord( family_catalog, house_name, 0 );

// TODO: error checking?

    // read in all names of the file until no more are extracted
    while ( read_result != EOF )
    {
      // case: there is more to add to the family (a partner, children)
      if ( read_result != '\n' )
      {
        // search the catalog for the newest head of household
        current_family_ndx = findHousehold( family_catalog, house_name ); 

        // complete the family of the current head of household
        read_result = completeFamily( input_file, family_tree,
                                      current_family_ndx );

        // add the children to the search catalog
        addChildrenToCatalog( family_catalog, family_tree, current_family_ndx );
      }

      // case: the end of the file wasn't read to
      if (  read_result != EOF )
      {
        // attempt to read in the next head of household
        read_result = getName( input_file, house_name );
      }
    }

    // close the file
    fclose( input_file );

    // deconstruct the search catalog
    destroyCatalog( family_catalog, 0 );
    free( family_catalog );

    // return the buffer dynamic memory
    free( house_name );
    house_name = NULL;
  }
  // otherwise, file opening failed and the error will be indicated by a nullptr

  // return the file reading success result
  return family_tree;
}


/**
  @var char_read   The character most recently read from the file stream. Is of
                   type int due to the fact that fgetc returns integers (to
                   accomodate the return of the EOF signal).
  @var ndx         Index for use in the buffer the_name provies.
*/
int getName( FILE* input_file, char* the_name )
{
  // variables
  int char_read = EOF; // makes use of fact that fgetc returns int
  int ndx = 0;

  // prime the reading loop
  char_read = fgetc( input_file );

  // case: EOF was not reached
  if ( char_read != EOF )
  {
    // read through any leading whitespace
    while ( isspace( char_read ) )
    {
      // get a new character
      char_read = fgetc( input_file );
    }

    // read in characters until either a whitespace one is found
    // or the end of the file is reached
    while ( !isspace( char_read ) && ( char_read != EOF ) )
    {
      // add the character to the name
      the_name[ndx] = char_read;

      // update the index
      ndx++;

      // get the next character
      char_read = fgetc( input_file );
    }

    // append a null terminator to the name
    the_name[ndx] = '\0';
  }

  // return the whitespace character immediately after the name (or EOF)
  return char_read;
}


/**
  @var new_family   A pointer used in the allocation and returning of a
                    newly created family node for use in the "first-born"
                    family tree. 
*/
Family* makeFamily( char* head_of_house )
{
  // variables
  Family* new_family = NULL;

  // dynamically allocate memory for the new family
  new_family = (Family*) malloc( sizeof(Family) );

  // case: allocation was successful
  if ( new_family != NULL )
  {
    // dynamically allocate memory for the names
    new_family->family_head_name = (char*) malloc( NAME_SIZE * sizeof( char ) );
      // TODO: error checking?

    // set the data members
    strcpy( new_family->family_head_name, head_of_house );
    new_family->partner = NULL;
  }

  // return the pointer to the memory block
  return new_family;
}


/**
  @var new_record       A pointer to the newly created FamilyRecord node.
  @var new_record_ndx   The location in the array/tree where the new
                        record belongs.
  @var compare_result   Used for the comparisons in the process of searching the
                        tree.
*/
FamilyRecord* addRecord( FamilyRecord** search_catalog, char* house_name, 
                         int family_tree_ndx )
{
  // variables
  FamilyRecord* new_record = NULL;
  int new_record_ndx = 0;
  int compare_result = 0;

  // locate the place in the search tree that the house should go into
  while ( search_catalog[new_record_ndx] != NULL )
  {
    // perform a name comparison with the current node
    compare_result = strcmp( house_name,
                             search_catalog[new_record_ndx]->family_head_name );

    // case: the new house name is less than the current one
    if( compare_result < 0 )
    {
      // follow the left sub-tree
      new_record_ndx = (2 * new_record_ndx) + 1;
    }
    // case: the new house name is greater than the current one
    else
    {
      // follow the right sub-tree
      new_record_ndx = (2 * new_record_ndx) + 2;
    }

    // TODO: shouldn't encounter name already in tree, implement error checking?
  }

  // dynamically allocate memory for the new record
  search_catalog[new_record_ndx] =
      (FamilyRecord*) malloc( sizeof( FamilyRecord ) );
  new_record = search_catalog[new_record_ndx];

  // case: allocation was successful
  if ( new_record != NULL )
  {
    // allocate memory for the house name
    new_record->family_head_name =
        (char*) calloc( NAME_SIZE, sizeof( char ) );

    // set the data members
    strcpy( new_record->family_head_name, house_name );
    new_record->family_ndx = family_tree_ndx;
  }

  // return the pointer to the newly added entry
  return new_record;
}


/**
  @var family_location   The index of the sought family in the "first-born"
                         tree.
  @var compare_result    Used in the searching of the binary search tree.
  @var ndx               The index of the current node in the tree being
                         considered for the search process.
*/
int findHousehold( FamilyRecord** search_catalog, char* house_name )
{
  // variables
  int family_location = NOT_FOUND;
  int compare_result = 1;
  int ndx = 0;

  // search for the sought family
  while ( ( compare_result != 0 ) )
  {
    // case: the current pointer is pointing to a valid record
    if ( ( search_catalog[ndx] != NULL ) && ( ndx < MAX_FAMILY_SIZE ) )
    {
      // compare the strings to see what the next action should be
      compare_result = strcmp( house_name,
                               search_catalog[ndx]->family_head_name );

      // case: the sought name is less than the one in the current record
      if ( compare_result < 0 )
      {
        // follow the left sub_tree
        ndx = (2 * ndx) + 1;
      }
      // case: the sought name is greater than that of the current record
      else if ( compare_result > 0 )
      {
        // follow the right sub-tree
        ndx = (2 * ndx) + 2;
      }
      // case: a match has been found
      else
      {
        // get the address of the desired family in the family tree
        family_location = search_catalog[ndx]->family_ndx;
      }
    }
    // case: the pointer is not pointing at a valid record
    else
    {
      // stop the loop
      compare_result = 0;
    }
  }

  // return the pointer to the family node in the Family tree
  return family_location;
}


/**
  @var read_result   The character most recently read from the file stream.
                     Used to indicate the end of a line or EOF.
  @var ndx           Indicates the index of where a child/sibling node will be
                     created in the "firs-born" tree.
  @var buffer        Used to hold names that are read in.
  @var parents       A pointer used to access the marriage partner's name.
*/
int completeFamily( FILE* input_file, Family** family_tree,
                     int current_family )
{
  // variables
  int read_result = EOF;
  int ndx = ((2 * current_family) + 1); // the first child's index in the tree
  char* buffer;
  Family* parents = family_tree[current_family];

  // dynamically allocate memory to the buffer and the partner name
  buffer = (char*) malloc( NAME_SIZE * sizeof( char ) );
  parents->partner = (char*) calloc( NAME_SIZE, sizeof( char ) );

  // read in the partner of the head of household
  read_result = getName( input_file, parents->partner );

  // read in children while the end of a line is not encountered
  while ( ( read_result != '\n' ) && ( read_result != EOF ) )
  {
    // read in the child's name
    read_result = getName( input_file, buffer );

    // create the child's node in the family tree
    family_tree[ndx] = makeFamily( buffer );

    // move on to the next child (which is a sibling of the current one)
    ndx = (2 * ndx) + 2;
  }

  // return the dynamic memory
  free( buffer );
  buffer = NULL;

  // return the most recent read result
  return read_result;
}


/**
  @var ndx   Used for traversing the "first-born" tree to locate all children
             of a given parent.
*/
void addChildrenToCatalog( FamilyRecord** search_catalog, Family** family_tree,
                           int parent_ndx )
{
  // variables
  int ndx = ((2 * parent_ndx) + 1);

  // add every child the family has to the search catalog
  while ( family_tree[ndx] != NULL )
  {
    // add the current child to the catalog
    addRecord( search_catalog, family_tree[ndx]->family_head_name,
               ndx );

    // move on to the next child of the generation (who is a sibiling of the
    // current one)
    ndx = (2 * ndx) + 2;
  }

  // no return - void
}


/**
  @var procedure_success   Used to signal a forking error.
  @var child_ndx           Used to locate the children of a given individual.
  @var child_generation    Indicates the generation of the current
                           individual (parent).
  @var next_child          Indicates the next child node whose data will be
                           printed.
  @var current_pid         The pid used for purposes of differentiating
                           forked processes.
*/
int printFamily( Family** family_tree, int parent_ndx, int generation )
{
  // variables
  int procedure_success = 0;
  int child_ndx = (2 * parent_ndx) + 1;
  int child_generation = generation + 1;
  Family* next_child = family_tree[child_ndx];
  pid_t current_pid = 0;

  // print the parent's name
  printParents( family_tree[parent_ndx], generation );

  // spawn and process all siblings
  while ( next_child != NULL )
  {
    // fork the process to get the children printed
    current_pid = fork();

    // case: the current process is the child process
    if ( current_pid == 0 )
    {
      // process the child and their family
      printFamily( family_tree, child_ndx, child_generation );

      // be sure to get the loop stopped, we don't want the parent cloning
      // itself and re-spawning children
      next_child = NULL;
    }
    // case: the current process is the parent one
    else if ( current_pid > 0 )
    {
      // wait for the child process to complete
      waitpid( current_pid, NULL, 0 );

      // move to the next child of the parent, aka the child's sibling
      // (follow right sub_tree)
      child_ndx = (2 * child_ndx) + 2;
      next_child = family_tree[child_ndx];
    }
    // case: there was an error in the forking
    else
    {
       // report the error
       printf( "!!! A forking error occurred, output incomplete. !!!\n" );

       // indicate the error to the program
       procedure_success = FORK_ERROR;

       // stop the loop
       next_child = NULL;
    }
  }

  // return the procedure success error code
  return procedure_success;
}


/**
  @var tab_num   The number of tabs already printed.
*/
void printParents( Family* the_family, int generation )
{
  // variables
  int tab_num = 0;

  // print a number of tabs equal to the generation number
  for ( tab_num = 0; tab_num < generation; tab_num++ )
  {
    // print a tab
    printf( "\t" );
  }

  // print the head of household's name
  printf( "%s", the_family->family_head_name );

  // print the parent process id number
  printf( "(%d)", getppid() );

  // case: there is a partner
  if ( the_family->partner != NULL )
  {
    // print the partner's name
    printf( "-%s", the_family->partner );
  }

  // print an endline
  printf( "\n" );

  // no return - void
}


/**
*/
void destroyCatalog( FamilyRecord** catalog, int current_record )
{
  // variables
    // none

  // case: we have not reached the end of a branch
  if ( catalog[current_record] != NULL )
  {
    // get rid of the left sub-tree
    destroyCatalog( catalog, ((2 * current_record) + 1) );

    // get rid of the right sub-tree
    destroyCatalog( catalog, ((2 * current_record) + 2) );

    // get rid of this record
    free( catalog[current_record]->family_head_name );
    free( catalog[current_record] );
    catalog[current_record] = NULL;
  }

  // no return - void
}


/**
*/
void genocide( Family** family_tree, int current_family )
{
  // variables
    // none

  // case: we have not reached the end of a family line
  if ( family_tree[current_family] != NULL )
  {
    // get rid of all of this family's children
    genocide( family_tree, ((2 * current_family) + 1) );

    // get rid of all of this family's siblings
    genocide( family_tree, ((2 * current_family) + 2) );

    // get rid of this family
    // free name data
    free( family_tree[current_family]->family_head_name );
    free( family_tree[current_family]->partner );
    free( family_tree[current_family] );
    family_tree[current_family] = NULL;
  }

  // no return - void
}

