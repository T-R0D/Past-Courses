/*******************************************************************************
*
*       This file is part of the General Hidden Markov Model Library,
*       GHMM version __VERSION__, see http://ghmm.org
*
*       Filename: ghmm/ghmm/mprintf.h
*       Authors:  Frank Nuebel
*
*       Copyright (C) 1998-2004 Alexander Schliep 
*       Copyright (C) 1998-2001 ZAIK/ZPR, Universitaet zu Koeln
*	Copyright (C) 2002-2004 Max-Planck-Institut fuer Molekulare Genetik, 
*                               Berlin
*                                   
*       Contact: schliep@ghmm.org             
*
*       This library is free software; you can redistribute it and/or
*       modify it under the terms of the GNU Library General Public
*       License as published by the Free Software Foundation; either
*       version 2 of the License, or (at your option) any later version.
*
*       This library is distributed in the hope that it will be useful,
*       but WITHOUT ANY WARRANTY; without even the implied warranty of
*       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*       Library General Public License for more details.
*
*       You should have received a copy of the GNU Library General Public
*       License along with this library; if not, write to the Free
*       Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
*
*
*       This file is version $Revision: 1436 $ 
*                       from $Date: 2005-10-12 07:35:29 -0400 (Wed, 12 Oct 2005) $
*             last change by $Author: grunau $.
*
*******************************************************************************/
/*******************************************************************************
  author       : Frank Nübel
  filename     : ghmm/ghmm/mprintf.h
  created      : TIME: 11:27:32     DATE: Wed 14. May 1997
  $Id: mprintf.h 1436 2005-10-12 11:35:29Z grunau $

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


*******************************************************************************/

#ifndef GHMM_MPRINTF_H
#define GHMM_MPRINTF_H

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/

/**
   @name Help functions for printing.
*/

  /**
   */
  char *ighmm_mprintf (char *dst, int maxlen, char *format, ...);
  /**
   */
  char *ighmm_mprintf_dyn (char *dst, int maxlen, char *format, ...);
  /**
   */
  char *ighmm_mprintf_va (char *dst, int maxlen, char *format, va_list args);
  /**
   */
  char *ighmm_mprintf_va_dyn (char *dst, int maxlen, char *format, va_list args);



#ifdef __cplusplus
}
#endif /*__cplusplus*/
#endif                          /* GHMM_MPRINTF_H */
