#  filename     : /zpr/bspk/src/hmm/ghmm/ghmm/Makefile.am
#  created      : DATE: April 2001
#  $Id: Makefile.am 2312 2013-06-21 16:13:13Z danzk87 $
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

SWIG_INTERFACE_FILES = ghmmwrapper.i \
                       wrapper_alphabet.i \
		       wrapper_arrays.i \
                       wrapper_cseq.i \
                       wrapper_dseq.i \
                       wrapper_dpseq.i \
                       wrapper_cmodel.i \
                       wrapper_dmodel.i \
                       wrapper_dpmodel.i \
                       wrapper_xmlfile.i

EXTENSION_C_FILES = gql.c \
                    pclasschange.c \
                    pclasschange.h \
                    sclass_change.c \
                    sclass_change.h \
                    sdclass_change.c \
                    sdclass_change.h

PYTHON_FILES = class_change.py \
               ghmmhelper.py \
               ghmm.py \
               ghmmunittests.py \
               mixture.py \
               modhmmer.py \
               setup.py

TESTDATA_FILES = testdata/discrete.xml \
                 testdata/multexon-4.xml \
                 testdata/d_seq.sqd \
                 testdata/tk.hmm \
                 testdata/multiple_hmmer.hmm \
                 testdata/tiny.smo \
                 testdata/xml_cont_multiple.xml \
                 testdata/test10.sqd \
                 testdata/test2.xml \
                 testdata/model_label.xml \
                 testdata/tiny.txt.sqd


EXTRA_DIST = ${SWIG_INTERFACE_FILES} ${EXTENSION_C_FILES} ${PYTHON_FILES} ${PYTHON_GATO_FILES} ${TESTDATA_FILES}

all: setup.py ghmm.py ${SWIG_INTERFACE_FILES}
	$(PYTHON) setup.py build

clean: setup.py
	$(PYTHON) setup.py clean
	rm -f message.txt

install: setup.py all
	$(PYTHON) setup.py install --prefix=${prefix}

check: ghmmunittests.py all ${TESTDATA_FILES}
	PYTHONPATH=./build/lib.linux-i686-$(PYTHONVER)/:$(PYTHONPATH) LD_LIBRARY_PATH=../ghmm/.libs:$(LD_LIBRARY_PATH) $(PYTHON) ghmmunittests.py

coverage: ghmmunittests.py all
	coverage.py -e
	coverage.py -x ghmmunittests.py
	coverage.py -r ghmm.py

distclean: setup.py
	$(PYTHON) setup.py clean --all
	rm -f message.txt

.PHONY: coverage
