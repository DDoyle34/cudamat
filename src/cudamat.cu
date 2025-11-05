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
    matrix.el = (float*)malloc(sizeof(float)*m*n);
    matrix.m = m;
    matrix.n = n;
    return matrix;
}

__global__ void set_const(float* a, float* k)
{
    a[blockIdx.x] = *k;
}

Matrix init_const(unsigned m, unsigned n, float k)
{
    Matrix a = init_empty(m, n);
    float *d_a, *d_k;
    cudaMalloc((void**)&d_a, sizeof(float)*m*n);
    cudaMalloc((void**)&d_k, sizeof(float));
    cudaMemcpy(d_k, &k, sizeof(float), cudaMemcpyHostToDevice);
    set_const<<<m*n,1>>>(d_a, d_k);
    cudaMemcpy(a.el, d_a, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_k);
    return a; 
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
    float* d_a = 0;
    cudaMalloc((void**)&d_a, sizeof(float)*m*n);
    curandGenerateUniform(gen, d_a, m*n);
    cudaMemcpy(a.el, d_a, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    return a;
}

void mfree(Matrix a)
{
    free(a.el);
}

__global__ void set_add(float* a, float* b, float* c)
{
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

Matrix madd(Matrix a, Matrix b)
{
    Matrix c = init_empty(a.m, a.n);
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(float)*a.m*a.n);
    cudaMalloc((void**)&d_b, sizeof(float)*b.m*b.n);
    cudaMalloc((void**)&d_c, sizeof(float)*c.m*c.n);
    cudaMemcpy(d_a, a.el, sizeof(float)*a.m*a.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.el, sizeof(float)*b.m*b.n, cudaMemcpyHostToDevice);
    set_add<<<a.m*a.n,1>>>(d_a, d_b, d_c);
    cudaMemcpy(c.el, d_c, sizeof(float)*c.m*c.n, cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return c;
}

Matrix mmult(Matrix a, Matrix b)
{
    Matrix c = init_zero(a.m, b.n);
    float alpha = 1.0;
    float *d_a, *d_b, *d_c, *d_k;
    cudaMalloc((void**)&d_a, sizeof(float)*a.m*a.n);
    cudaMalloc((void**)&d_b, sizeof(float)*b.m*b.n);
    cudaMalloc((void**)&d_c, sizeof(float)*c.m*c.n);
    cudaMalloc((void**)&d_k, sizeof(float));
    cudaMemcpy(d_a, a.el, sizeof(float)*a.m*a.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.el, sizeof(float)*b.m*b.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, &alpha, sizeof(float), cudaMemcpyHostToDevice);
    cublasSgemm(get_handle(), CUBLAS_OP_N, CUBLAS_OP_N, a.m, b.n, a.n, d_k, 
            d_a, a.m, d_b, b.m, d_k, d_c, c.m);
    cudaMemcpy(c.el, d_c, sizeof(float)*b.m*b.n, cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_k);
    return c;
}

Matrix kmult(float k, Matrix a)
{
    Matrix c = init_zero(a.m, a.n);
    float *d_a, *d_c, *d_k;
    cudaMalloc((void**)&d_a, sizeof(float)*a.m*a.n);
    cudaMalloc((void**)&d_k, sizeof(float));
    cudaMalloc((void**)&d_c, sizeof(float)*c.m*c.n);
    cudaMemcpy(d_a, a.el, sizeof(float)*a.m*a.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, &k, sizeof(float), cudaMemcpyHostToDevice);
    cublasSaxpy(get_handle(), a.m * a.n, d_k, d_a, 1, d_c, 1);
    cudaMemcpy(c.el, d_c, sizeof(float)*c.m*c.n, cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_k); cudaFree(d_c);
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
    Matrix b = init_empty(a.m, a.n);
    cublasScopy(get_handle(), a.m * a.n, a.el, 1, b.el, 1);
    return b;
}
