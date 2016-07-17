#  author       : Achim Gaedke
#  filename     : ghmm/tests/Makefile.am
#  created      : DATE: April 2001
#  $Id: Makefile.am 2314 2013-06-24 15:06:51Z ejb177 $
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

SUBDIRS = data

# tests and examples

CLEANFILES = ghmm.log ghmm.sum site.exp site.bak
MAINTAINERCLEANFILES = Makefile.in
# stuff from dejagnu

INCLUDES=-I$(top_srcdir)

#these tests will not be installed
# in progress: test_sdfoba nullmodel ciscreen test_sdmodel 
noinst_PROGRAMS = randvar_test \
                  root_finder_test \
                  coin_toss_test \
                  two_states_three_symbols \
                  libxml-test \
                  chmm \
                  chmm_test \
                  shmm_viterbi_test \
                  test_gsl_ran_gaussian_tail \
                  sequences_test \
                  sequences_old_format \
                  label_higher_order_test \
		  read_fa \
                  mcmc

LDADD = $(top_builddir)/ghmm/.libs/libghmm.a

TESTS_ENVIRONMENT = GHMM_SILENT_TESTS
TESTS =           root_finder_test \
		  coin_toss_test \
		  two_states_three_symbols \
		  libxml-test \
		  chmm \
		  chmm_test \
		  shmm_viterbi_test \
		  test_gsl_ran_gaussian_tail \
		  sequences_test \
		  sequences_old_format \
		  label_higher_order_test \
		  read_fa \
                  mcmc
