/*
  author       : David Posada <dposada@variagenics.com>
  filename     : ghmm/tests/sequences_old_format.c
  created      : DATE: Februar 2002
  $Id: sequences_old_format.c 1451 2005-10-18 10:21:55Z grunau $
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
  
*/

#ifdef HAVE_CONFIG_H
#include "../config.h"
#endif /* HAVE_CONFIG_H */

/* should be corrected with ghmm/sequence.c version 1.9 */
#include <stdio.h>
#include <ghmm/sequence.h>

#include <ghmm/obsolete.h>
   
int main()
{
#ifdef GHMM_OBSOLETE
    int test_result=0;
    const char* double_sequences_file="data/test100.sqd";
    ghmm_cseq **sqd = NULL;
    int sqd_number;
    const char* int_sequences_file="data/sequences_old_format.sq";    
    ghmm_dseq **data = NULL;
    int data_number;

    /* read double sequences (this works fine)*/
    fprintf(stderr,"reading double sequences from %s ...",double_sequences_file);
    sqd=ghmm_cseq_read((char*)double_sequences_file, &sqd_number);
    if (sqd==NULL) {
      test_result=1;
      fprintf(stdout, " Failed\n");
    }
    else {
      fprintf(stdout," Done\n");
      ghmm_cseq_free(sqd);
    }


     /* read int sequences (this gives a segmentation fault)*/
    fprintf(stderr,"reading int sequences from %s ...",int_sequences_file);
    data=ghmm_dseq_read((char*)int_sequences_file,&data_number);
    if (data==NULL) {
      test_result=1;
      fprintf(stdout, " Failed\n");
    }
    else {
      fprintf(stdout," Done\n");
      ghmm_dseq_free(data);
    }
    return test_result;
#else /* GHMM_OBSOLETE */
    return 0;
#endif /* GHMM_OBSOLETE */
}
