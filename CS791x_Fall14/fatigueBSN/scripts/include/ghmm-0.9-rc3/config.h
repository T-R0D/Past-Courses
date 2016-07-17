/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.in by autoheader.  */

/* use matrix operations from lapack */
/* #undef DO_WITH_ATLAS */

/* use GSL functions instead of ghmm equivalents */
/* #undef DO_WITH_GSL */

/* use hack to estimate only diagonal covariance matrix */
/* #undef DO_WITH_GSL_DIAGONAL_HACK */

/* gsl_histogram_set_ranges_uniform is defined */
/* #undef GSL_HISTOGRAM_SET_RANGES_UNIFORM */

/* root solver allocation takes only one argument */
/* #undef GSL_ROOT_FSLOVER_ALLOC_WITH_ONE_ARG */

/* clapack_dgetrf exists */
/* #undef HAVE_CLAPACK_DGETRF */

/* clapack_dgetri exists */
/* #undef HAVE_CLAPACK_DGETRI */

/* clapack_dpotrf exists */
/* #undef HAVE_CLAPACK_DPOTRF */

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* struct gsl_interval exists */
/* #undef HAVE_GSL_INTERVAL */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the `bsd' library (-lbsd). */
/* #undef HAVE_LIBBSD */

/* Define to 1 if you have the `m' library (-lm). */
#define HAVE_LIBM 1

/* Define to 1 if you have the `pthread' library (-lpthread). */
#define HAVE_LIBPTHREAD 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "ghmm"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT ""

/* Define to the full name of this package. */
#define PACKAGE_NAME "ghmm"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "ghmm 0.9-rc3"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "ghmm"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "0.9-rc3"

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
#define TIME_WITH_SYS_TIME 1

/* Define to 1 if your <sys/time.h> declares `struct tm'. */
/* #undef TM_IN_SYS_TIME */

/* Version number of package */
#define VERSION "0.9-rc3"

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Define to `unsigned int' if <sys/types.h> does not define. */
/* #undef size_t */
