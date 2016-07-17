#  author       : Achim Gaedke
#  filename     : ghmm/Makefile.am
#  created      : DATE: April 2001
#  $Id: Makefile.am 2270 2009-04-27 14:29:38Z grunau $
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

SUBDIRS = @SUBDIRS@

MAINTAINERCLEANFILES = \
aclocal.m4 \
configure \
Makefile.in \
stamp-h.in \
config.h.in \
mkinstalldirs \
missing \
install-sh \
INSTALL \
COPYING \
$(distdir).tar.gz -r $(distdir) \
ltmain.sh \
config.guess \
config.sub \
$(distdir).tar.gz \
-r $(distdir)

pkgdata_DATA = doc/ghmm.dtd.1.0

EXTRA_DIST = \
COPYRIGHT.txt \
GPL.txt \
LGPL.txt \
ToDo \
autogen.sh \
doc/xml_example.xml \
doc/xml_cont_example.xml \
doc/xml_mix_cont_example.xml \
doc/xml_switching-example.xml \
doc/xml_data_structure.fig \
${pkgdata_DATA}

doc: doc/ghmm/Doxyfile doc/ghmmwrapper/Doxyfile
	@DOXYGEN@ doc/ghmm/Doxyfile
	@DOXYGEN@ doc/ghmmwrapper/Doxyfile

doc-upload: doc
	rsync -avP -e ssh --delete doc/ghmm/html/ grunau,ghmm@web.sourceforge.net:htdocs/doc/ghmm
	rsync -avP -e ssh --delete doc/ghmmwrapper/html/ grunau,ghmm@web.sourceforge.net:htdocs/doc/ghmmwrapper

.PHONY: doc doc-upload
