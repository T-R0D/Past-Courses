#ifndef GHMM_CFBGIBBS_H
#define GHMM_CFBGIBBS_H

#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif


/* runs the forward backward gibbs burnIn times
 * mo: model
 * seed: seed 
 * obs: observation
 * totalobs: length of observation sequence
 * pA: prior count for A
 * pB: prior count for B
 * pPi: prior count for pi
 * R: length of compression
 * burnIn: number of times to run forward backward gibbs 
 * return int* state path
 */
int** ghmm_dmodel_cfbgibbs (ghmm_dmodel *mo, ghmm_dseq* seq, double **pA, double **pB, double *pPi, int R, int burnIn, int seed);


#ifdef __cplusplus
}
#endif

#endif
