#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include "cuda.h"
#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define SUBMATRIX_SIZE 10000
#define BLOCK_SIZE 16

float getnum() {
  return ((float) rand())/((float) RAND_MAX);
}


__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
  __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  // Identify the row and column of the Pd element to work on
  int Row = by * BLOCK_SIZE + ty;
  int Col = bx * BLOCK_SIZE + tx;
  float Pvalue = 0;
  // Loop over the Md and Nd tiles required to compute the Pd element
  for (int m = 0; m < Width/BLOCK_SIZE; ++m) {
    // Collaborative loading of Md and Nd tiles into shared memory
    Mds[ty][tx] = Md[Row*Width + (m*BLOCK_SIZE + tx)];
    Nds[ty][tx] = Nd[Col + (m*BLOCK_SIZE + ty)*Width];
    __syncthreads();
    for (int k = 0; k < BLOCK_SIZE; ++k)
      Pvalue += Mds[ty][k] * Nds[k][tx];
    __syncthreads();
  }
  Pd[Row*Width+Col] = Pvalue;
}


void make_identity(float* a, int startx, int starty, int length, int n) {
  // fill a with the identity from start to end
  int i, j;
  for (i=startx; i<length+startx; i++) {
    for (j=starty; j<length+starty; j++) {
      a[i*n+j] = (i-startx==j-starty) ? 1.0f : 0.0f;
    }
  }
}

void make_negidentity(float* a, int startx, int starty, int length, int n) {
  // fill a with the identity from start to end
  int i, j;
  for (i=startx; i<length+startx; i++) {
    for (j=starty; j<length+starty; j++) {
      a[i*n+j] = (i-startx==j-starty) ? -1.0f : 0.0f;
    }
  }
}

void make_x(float* x, int length) {
  int i, j;
  for (i=0; i<length; i++) {
    for (j=0; j<length; j++) {
      x[i*length+j] = getnum();
      //x[i*length+j] = 0.3;
    }
  }
}

void make_zero(float* a, int startx, int starty, int length, int n) {
  int i, j;
  for (i=startx; i<length+startx; i++) {
    for (j=starty; j<length+starty; j++) {
      a[i*n+j] = 0.0f;
    }
  }
}

void copy_x(float* a, float* x, int startx, int starty, int length, int n) {
  int i, j;
  for (i=startx; i<length+startx; i++) {
    for (j=starty; j<length+starty; j++) {
      a[i*n+j] = x[(i-startx)*length+(j-starty)];
    }
  }
}

void copy_2x(float* a, float* x, int startx, int starty, int length, int n) {
  int i, j;
  for (i=startx; i<length+startx; i++) {
    for (j=starty; j<length+starty; j++) {
      a[i*n+j] = 2*x[(i-startx)*length+(j-starty)];
    }
  }
}

void copy_negx(float* a, float* x, int startx, int starty, int length, int n) {
  int i, j;
  for (i=startx; i<length+startx; i++) {
    for (j=starty; j<length+starty; j++) {
      a[i*n+j] = (-1)*x[(i-startx)*length+(j-starty)];
    }
  }
}

void make_result(float* a, int length) {
  int i, j;
  int half = length>>1;
  for (i=0; i<length; i++) {
    for (j=0; j<length; j++) {
      if (i == j) {
	if (i>=half)
	  a[i*length+j] = -(1.0f);
	else
	  a[i*length+j] = 1.0f;
      }
      else
	a[i*length+j] = 0.0f;
    }
  }
}

#define ffabs(val) (val) < 0.0f ? (-(val)) : (val)

float rothVerf(float* a, float* b, int n) {
  float sum = 0;
  int i, j;
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      sum += ffabs(a[i*n+j] - b[i*n+j]);
    }
  }
  return sum;
}

void print_mat(float* a, int n) {
  int i, j;
  if (n<64) {
    for (i=0; i<n; i++) {
      for (j=0; j<n; j++) {
	printf("%.3f\t", a[i*n+j]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

float trace(float* a, int n) {
  int i;
  float total = 1.0f;
  for (i=0; i<n; i++) {
    total *= a[i*n+i];
  }
  return total;
}


int main() {
  srand(100);
  int n = 2*SUBMATRIX_SIZE;
  int half = SUBMATRIX_SIZE;
  size_t totalsize = sizeof(float)*n*n;
  size_t halfsize = sizeof(float)*half*half;
  float *x, *a, *b, *c, *d;

  cudaMallocHost((void**) &a, totalsize);
  cudaMallocHost((void**) &b, totalsize);
  cudaMallocHost((void**) &c, totalsize);
  cudaMallocHost((void**) &d, totalsize);
  cudaMallocHost((void**) &x, halfsize);
  
  if ((x==NULL) || (a==NULL) || (b==NULL) || (c==NULL) ||
      (d==NULL)) {
    printf("Matrix allocation error on host\n");
    exit(1);
  }

  make_x(x, half);
  print_mat(x, half);

  // construct first matrix
  make_identity(a, 0, 0, half, n);
  copy_x(a, x, 0, half, half, n);
  make_zero(a, half, 0, half, n);
  make_identity(a, half, half, half, n);
  printf("Trace of a: %f\n", trace(a, n));

  // second matrix
  make_identity(b, 0, 0, half, n);
  copy_2x(b, x, 0, half, half, n);
  make_zero(b, half, 0, half, n);
  make_negidentity(b, half, half, half, n);

  // third
  make_identity(c, 0, 0, half, n);
  copy_negx(c, x, 0, half, half, n);
  make_zero(c, half, 0, half, n);
  make_identity(c, half, half, half, n);

  // result
  make_result(d, n);
  print_mat(a, n);
  print_mat(b, n);
  print_mat(c, n);
  print_mat(d, n);

  // allocate on device
  float *dev_a, *dev_b, *dev_c, *dev_inter;
  cudaMalloc((void**) &dev_a, totalsize);
  cudaMalloc((void**) &dev_b, totalsize);
  cudaMalloc((void**) &dev_c, totalsize);
  cudaMalloc((void**) &dev_inter, totalsize);
  
  // copy to device
  cudaMemcpy(dev_a, a, totalsize, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, totalsize, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, totalsize, cudaMemcpyHostToDevice);

  unsigned int grid_rows = n / BLOCK_SIZE;
  unsigned int grid_cols = n / BLOCK_SIZE;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  // intermediate matrix product
  //MatrixMulKernel<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_inter, n);
  cublasHandle_t handle;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, dev_a, n, dev_b, n, &beta, dev_inter, n);
  cudaThreadSynchronize();

  // reuse old matrix
  //MatrixMulKernel<<<dimGrid, dimBlock>>>(dev_inter, dev_c, dev_a, n);
  cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, dev_inter, n, dev_c, n, &beta, dev_a, n);

  // bring product back to cpu
  cudaMemcpy(a, dev_a, totalsize, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();

  // check a against the result d
  float sum = rothVerf(a, d, n);
  printf("Total Error: %f\n", sum);
  print_mat(a, n);

  // cleanup and exit
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  cudaFree(dev_inter);
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  cudaFreeHost(d);
  cudaFreeHost(x);
  return 0;
}
