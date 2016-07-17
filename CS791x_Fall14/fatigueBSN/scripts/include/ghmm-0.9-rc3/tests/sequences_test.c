/*******************************************************************************
  author       : Achim Gädke
  filename     : ghmm/tests/sequences_test.c
  created      : DATE: Thu 26. June 2001
  $Id: sequences_test.c 2193 2008-03-14 10:48:53Z grunau $

Copyright (C) 1998-2005 Alexander Schliep
Copyright (C) 1998-2001 ZAIK/ZPR, Universitaet zu Koeln
Copyright (C) 2002-2005 Max-Planck-Institut fuer Molekulare Genetik, Berlin

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Library General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Library General Public License for more details.

You should have received a copy of the GNU Library General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA




*******************************************************************************
Parts of the library are Copyright of Sun Microsystems, Inc.
and re distributed under following license

 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================


*****************************************************************************/

#ifdef HAVE_CONFIG_H
#include "../config.h"
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <stdlib.h>
#include <ghmm/sequence.h>

void sequence_alloc_print(void)
{
  ghmm_dseq* seq_array;
  int i;

  seq_array= ghmm_dseq_calloc(1);
  seq_array->seq_len[0]=10;
#ifdef GHMM_OBSOLETE
  seq_array->seq_label[0]=100;
#endif /* GHMM_OBSOLETE */
  seq_array->seq_id[0]=101.0;
  seq_array->seq[0]=(int*)malloc(seq_array->seq_len[0]*sizeof(int));

  for (i=0; i<seq_array->seq_len[0]; i++)
    seq_array->seq[0][i]=1;

  ghmm_dseq_print_xml(seq_array, stdout);

  ghmm_dseq_free(&seq_array);
}

int main()
{
  sequence_alloc_print();
  return 0;
}
