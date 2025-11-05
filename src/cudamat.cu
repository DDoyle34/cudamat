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

__global__ void set_mult(float* a, float* b, float* c)
{
}

Matrix mmult(Matrix a, Matrix b)
{
    Matrix c = init_zero(a.m, b.n);
    float alpha = 1.0;
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(float)*a.m*a.n);
    cudaMalloc((void**)&d_b, sizeof(float)*b.m*b.n);
    cudaMalloc((void**)&d_c, sizeof(float)*c.m*c.n);
    cublasSetMatrix(a.m, a.n, sizeof(float), a.el, a.m, d_a, a.m);
    cublasSetMatrix(b.m, b.n, sizeof(float), b.el, b.m, d_b, b.m);
    cublasSgemm(get_handle(), CUBLAS_OP_N, CUBLAS_OP_N, a.m, b.n, a.n, &alpha, 
            d_a, a.m, d_b, b.m, &alpha, d_c, c.m);
    cublasGetMatrix(a.m, a.n, sizeof(float), d_c, c.m, c.el, c.m);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return c;
}

Matrix kmult(float k, Matrix a)
{
    Matrix c = init_zero(a.m, a.n);
    float *d_a, *d_c;
    cudaMalloc((void**)&d_a, sizeof(float)*a.m*a.n);
    cudaMalloc((void**)&d_c, sizeof(float)*c.m*c.n);
    cudaMemcpy(d_a, a.el, sizeof(float)*a.m*a.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.el, sizeof(float)*c.m*c.n, cudaMemcpyHostToDevice);
    cublasSaxpy(get_handle(), a.m * a.n, &k, d_a, 1, d_c, 1);
    cudaMemcpy(c.el, d_c, sizeof(float)*c.m*c.n, cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_c);
    return c;
}

__global__ void set_trans(float* a, float* c, unsigned* m, unsigned* n)
{
    c[blockIdx.x] = a[(blockIdx.x % *n) * (*m) + (blockIdx.x)/(*m)];
}

Matrix mtrans(Matrix a)
{
    Matrix c = init_zero(a.n, a.m);
    Matrix I = ident(a.m);
    unsigned *d_m, *d_n;
    float *d_a, *d_c;
    cudaMalloc((void**)&d_a, sizeof(float)*a.m*a.n);
    cudaMalloc((void**)&d_m, sizeof(unsigned));
    cudaMalloc((void**)&d_n, sizeof(unsigned));
    cudaMalloc((void**)&d_c, sizeof(float)*c.n*c.m);
    cudaMemcpy(d_a, a.el, sizeof(float)*a.m*a.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, &(a.m), sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, &(a.n), sizeof(unsigned), cudaMemcpyHostToDevice);
    set_trans<<<a.m*a.n,1>>>(d_a, d_c, d_m, d_n);
    cudaMemcpy(c.el, d_c, sizeof(float)*c.m*c.n, cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_c); cudaFree(d_m); cudaFree(d_n);
    return c;
}

__global__ void set_sigmoid(float* a)
{
    a[blockIdx.x] = 1 / (1 + exp((float)(-1) * a[blockIdx.x])); 
}

Matrix msigmoid(Matrix a)
{
    Matrix c = init_empty(a.m, a.n);
    float* d_c;
    cudaMalloc((void**)&d_c, sizeof(float)*a.m*a.n);
    cudaMemcpy(d_c, a.el, sizeof(float)*a.m*a.n, cudaMemcpyHostToDevice);
    set_sigmoid<<<a.m*a.n,1>>>(d_c);
    cudaMemcpy(c.el, d_c, sizeof(float)*c.m*c.n, cudaMemcpyDeviceToHost);
    cudaFree(d_c);
    return c;
}

__global__ void set_ident(float* a, unsigned* m)
{
    a[blockIdx.x * (*m + 1)] = 1;
}

Matrix ident(unsigned m)
{
    Matrix a = init_zero(m, m);
    float *d_a;
    unsigned *d_m;
    cudaMalloc((void**)&d_a, sizeof(float)*m*m);
    cudaMalloc((void**)&d_m, sizeof(unsigned));
    cudaMemcpy(d_a, a.el, sizeof(float)*m*m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, &m, sizeof(float), cudaMemcpyHostToDevice);
    set_ident<<<m*m,1>>>(d_a, d_m);
    cudaMemcpy(a.el, d_a, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_m);
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
