// Matrix based operations using CUBLAS
#include "cudamat.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdbool.h>
#include <curand.h>
#include <curand_kernel.h>

static cublasHandle_t get_handle()
{
    static bool active = false; 
    static cublasHandle_t handle; 
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
    matrix.el = 0;
    cudaMalloc((void**)&(matrix.el), sizeof(float)*m*n);
    matrix.m = m;
    matrix.n = n;
    return matrix;
}

Matrix init_const(unsigned m, unsigned n, float k)
{
    Matrix matrix = init_empty(m, n);
    float* h_k = (float*)malloc(sizeof(float)*m*n);
    for (unsigned i = 0; i < m*n; i++) {
        h_k[i] = k;
    }
    cublasSetMatrix(m, n, sizeof(matrix.el[0]), h_k, m, matrix.el, m);
    free(h_k);
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
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock());
    curandGenerateUniform(gen, a.el, m * n);
    //const float scale = 2.0;
    //const float shift = -1.0;
    //cublasSetMatrix(m, n, sizeof(a.el[0]), h_r, m, a.el, m);
    return a;
}

void mfree(Matrix a)
{
    cudaFree(a.el);
}

Matrix madd(Matrix a, Matrix b)
{
    Matrix c = init_zero(a.m, a.n);
    float alpha = 1.0; 
    cublasScopy(get_handle(), b.m * b.n, b.el, 1, c.el, 1);
    cublasSaxpy(get_handle(), a.m * a.n, &alpha, a.el, 1, c.el, 1);
    return c;
}

Matrix mmult(Matrix a, Matrix b)
{
    Matrix c = init_zero(a.m, b.n);
    float alpha = 1.0;
    cublasSgemm(get_handle(), CUBLAS_OP_N, CUBLAS_OP_N, a.m, b.n, a.n, &alpha, 
            a.el, a.m, b.el, b.m, &alpha, c.el, c.m);
    return c;
}

Matrix kmult(float k, Matrix a)
{
    Matrix c = init_zero(a.m, a.n);
    cublasSaxpy(get_handle(), a.m * a.n, &k, a.el, 1, c.el, 1);
    return c;
}

Matrix mtrans(Matrix a)
{
    Matrix c = init_zero(a.n, a.m);
    Matrix I = ident(a.m);
    float alpha = 1.0;
    cublasSgemm(get_handle(), CUBLAS_OP_N, CUBLAS_OP_T, I.m, a.n, I.m, &alpha, 
            I.el, I.m, a.el, a.m, &alpha, c.el, c.m);
    return c;
}

Matrix msigmoid(Matrix a)
{
    Matrix c = init_empty(a.m, a.n);
    float* h_a = (float*)malloc(sizeof(float) * a.m * a.n);
    cublasGetMatrix(a.m, a.n, sizeof(a.el[0]), a.el, a.m, h_a, a.m);
    for (unsigned i = 0; i < a.m; i++) {
        h_a[i] = 1 / (1 + exp((float)(-1) * h_a[i]));
    }
    cublasSetMatrix(c.m, c.n, sizeof(h_a[0]), h_a, c.m, c.el, c.m);
    free(h_a);
    return c;
}

Matrix ident(unsigned m)
{
    Matrix a = init_zero(m, m);
    float* h_a = (float*)malloc(sizeof(float) * a.m * a.n);
    for (unsigned i = 0; i < m; i++) {
        h_a[IDX2C(i, i, m)] = 1;
    }
    cublasSetMatrix(a.m, a.n, sizeof(h_a[0]), h_a, a.m, a.el, a.m);
    free(h_a);
    return a;
}

void mprint(Matrix a)
{
    float* h_a = (float*)malloc(sizeof(float)*(a.m)*(a.n));
    cublasGetMatrix(a.m, a.n, sizeof(a.el[0]), a.el, a.m, h_a, a.m);
    printf("{ ...\n");
    for (unsigned i = 0; i < a.m; i++) {
        printf("\t{");
        for (unsigned j = 0; j < a.n; j++) {
            printf(" %f ", h_a[IDX2C(i, j, a.m)]);
        }
        printf("}\n");
    }
    printf("... }\n");
    free(h_a);
}

Matrix mcopy(Matrix a)
{
    Matrix b = init_empty(a.m, a.n);
    cublasScopy(get_handle(), a.m * a.n, a.el, 1, b.el, 1);
    return b;
}
