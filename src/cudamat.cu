// Matrix based operations using CUBLAS
#include "cudamat.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdbool.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static cublasHandle_t get_handle()
{
    static bool active = false; 
    cublasHandle_t handle; 
    if (!active) {
        active = true; 
        cublasCreate(&handle);
    }
    return handle;
}

void unhandle() 
{
    cublasDestroy(get_handle());
}

Matrix init_empty(unsigned m, unsigned n)
{
    Matrix matrix;
    matrix.el = (float*)malloc(sizeof(float) * m * n);
    matrix.m = m;
    matrix.n = n;
    return matrix;
}

Matrix init_const(unsigned m, unsigned n, float k)
{
    Matrix matrix = init_empty(m, n);
    for (unsigned i = 0; i < m; i++) {
        matrix.el[i] = k;
    }
    return matrix; 
}

Matrix init_zero(unsigned m, unsigned n)
{
    return init_const(m, n, 0);
}

Matrix init_ones(unsigned m, unsigned n)
{
    return init_const(m, n, 1);
}

Matrix init_rand(unsigned m, unsigned n)
{
    Matrix a = init_empty(m, n);
    for (unsigned i = 0; i < m; i++) {
        a.el[i] = (float)2 * (rand() - (float)(RAND_MAX / 2)) / (float)RAND_MAX;
    }
    return a;
}

void mfree(Matrix a)
{
    free(a.el);
}

Matrix madd(Matrix a, Matrix b)
{
    if (a.m != b.m || a.n != b.n) {
        perror("Matrix size mismatch!\n");
        return a;
    }
    Matrix c = init_empty(a.m, a.n);
    float* d_a = 0;
    float* d_b = 0;
    float* d_c = 0;
    float alpha = 1.0; 
    cudaMalloc((void**)&d_a, a.m * a.n * sizeof(d_a[0]));
    cudaMalloc((void**)&d_b, b.m * b.n * sizeof(d_b[0]));
    cudaMalloc((void**)&d_c, c.m * c.n * sizeof(d_c[0]));
    cublasSetMatrix(a.m, a.n, sizeof(a.el[0]), a.el, a.m, d_a, a.m);
    cublasSetMatrix(a.m, a.n, sizeof(a.el[0]), a.el, a.m, d_a, a.m);
    cublasSetMatrix(a.m, a.n, sizeof(a.el[0]), a.el, a.m, d_a, a.m);
    cublasScopy(get_handle(), b.m * b.n, d_b, 1, d_c, 1);
    cublasSaxpy(get_handle(), a.m * a.n, &alpha, d_a, 1, d_c, 1);
    cublasGetMatrix(c.m, c.n, sizeof(a.el[0]), d_c, c.m, c.el, c.m);
    return c;
}

Matrix mmult(Matrix a, Matrix b)
{
    if (a.n != b.m) {
        perror("Matrix size mismatch!\n");
        return a;
    }

    Matrix c = init_zero(a.m, b.n);
    
    return c;
}

Matrix kmult(float k, Matrix a)
{
    Matrix c; 
    c.m = a.m;
    c.n = a.n;
    return c;
}

Matrix mtrans(Matrix a)
{
    Matrix c;
    c.m = a.m;
    c.n = a.n;
    return c;
}

Matrix msigmoid(Matrix a)
{
    Matrix c;
    c.m = a.m;
    c.n = a.n;
    for (unsigned i = 0; i < a.m; i++) {
        c.el[i] = 1 / (1 + exp((float)(-1) * a.el[i]));
    }
    return c;
}

Matrix ident(unsigned m)
{
    Matrix a = init_zero(m, m);
    for (unsigned i = 0; i < m; i++) {
        a.el[IDX2C(i, i, m)] = 1;
    }
    return a;
}

void mprint(Matrix a)
{
    printf("{ ...\n");
    for (unsigned i = 0; i < a.m; i++) {
        printf("\t{");
        for (unsigned j = 0; j < a.n; j++) {
            printf(" %f ", a.el[IDX2C(i, j, a.m)]);
        }
        printf("}\n");
    }
    printf("... }\n");
}

Matrix mcopy(Matrix a)
{
    Matrix b;
    b.el = (float*)malloc(sizeof(float) * a.m * a.n);
    b.m = a.m;
    b.n = a.n;
    return b;
}

Matrix ginvert(Matrix a)
{
    Matrix b = mcopy(a);
    Matrix c = ident(a.m);

    return c;
}
