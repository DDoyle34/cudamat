#ifndef CUDAMAT_H
#define CUDAMAT_H

#define IDX2C(i,j,ld) (((j)*(ld))+(i))       

typedef struct {
    float* el;
    unsigned m;
    unsigned n;
} Matrix;

Matrix init_const(unsigned m, unsigned n, float k);

Matrix init_zero(unsigned m, unsigned n);

Matrix init_ones(unsigned m, unsigned n);

Matrix init_rand(unsigned m, unsigned n);

void mfree(Matrix a);

Matrix madd(Matrix a, Matrix b);

Matrix mmult(Matrix a, Matrix b);

Matrix kmult(float k, Matrix a);

Matrix mtrans(Matrix a);

Matrix msigmoid(Matrix a);

Matrix ident(unsigned m);

void mprint(Matrix a);

Matrix ginvert(Matrix a);

#endif
