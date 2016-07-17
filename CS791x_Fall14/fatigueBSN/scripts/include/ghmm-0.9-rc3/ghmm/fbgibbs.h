#ifndef GHMM_FBGIBBS_H
#define GHMM_FBGIBBS_H

#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif
//**uses gsl**

int sample(int seed, double* dist, int N);

void init_priors(ghmm_dmodel *mo, double ***pA, double ***pB, double **pPi);

void update(ghmm_dmodel* mo, double **transistions, double *obsinstate, double **obsinstatealpha);

void updateH(ghmm_dmodel* mo,double **transistions, double *obsinstate, double **obsinstatealpha);

int** ghmm_dmodel_fbgibbs(ghmm_dmodel * mo, ghmm_dseq*  seq, double **pA, double **pB, double *pPi, int burnIn, int seed);





#ifdef __cplusplus
}
#endif
#endif
