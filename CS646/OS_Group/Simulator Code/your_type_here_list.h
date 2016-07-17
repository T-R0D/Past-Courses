#ifndef __your_type_here_list.h
#define __your_type_here_list.h

typedef struct
{
  your_type_here* list_head;
  your_type_here* list_tail;
  your_type_here* cursor;
  your_type_here* next;
  your_type_here* previous;
} your_type_hereList;


void your_type_here_list_init( your_type_hereList* the_list );

void your_type_here_list_clear( your_type_hereList* the_list );

void your_type_here_list_push( your_type_hereList* the_list,
                              your_type_here* new_item );

void your_type_here_list_insert( your_type_hereList* the_list,
                                 your_type_here* new_item );

your_type_here* your_type_here_list_pop( your_type_hereList* the_list );

your_type_here* your_type_here_list_delete( your_type_hereList* the_list );

int your_type_here_list_isEmpty( your_type_hereList* the_list );

#endif
