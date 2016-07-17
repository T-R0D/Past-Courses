#  author       : Achim Gaedke
#  filename     : /zpr/bspk/src/hmm/ghmm/tools/Makefile.am
#  created      : DATE: April 2001
#  $Id: Makefile.am 2238 2008-10-09 14:43:35Z grunau $
#
# Copyright (C) 1998-2005 Alexander Schliep
# Copyright (C) 1998-2001 ZAIK/ZPR, Universitaet zu Koeln
# Copyright (C) 2002-2005 Max-Planck-Institut fuer Molekulare Genetik, Berlin
# 
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
# 
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
# 
# 
# 
# 
# *******************************************************************************
# Parts of the library are Copyright of Sun Microsystems, Inc.
# and re distributed under following license
# 
#  * ====================================================
#  * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
#  *
#  * Developed at SunPro, a Sun Microsystems, Inc. business.
#  * Permission to use, copy, modify, and distribute this
#  * software is freely granted, provided that this notice
#  * is preserved.
#  * ====================================================
# 

include $(top_srcdir)/dist_hook.mak

SUBDIRS = .
DISTCLEANFILES = *~
MAINTAINERCLEANFILES = Makefile.in

BUILT_SOURCES = 
INCLUDES = -I$(top_srcdir)

bin_PROGRAMS = probdist cluster scluster smix_hmm $(OBSOLETE_TOOLS)
EXTRA_PROGRAMS = smo2xml

probdist_SORUCES = probdist.c
cluster_SOURCES  = cluster.c
scluster_SOURCES = scluster.c
smix_hmm_SOURCES = smix_hmm.c
smo2xml_SOURCES = smo2xml.c

LDADD = $(top_builddir)/ghmm/.libs/libghmm.a
bin_SCRIPTS = ghmm-config
