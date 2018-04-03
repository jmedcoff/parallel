#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 16

float getnum() {
  return rand()/((float) RAND_MAX);
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
	  a[i*length+j] = -1.0f;
	else
	  a[i*length+j] = 1.0f;
      }
      else
	a[i*length+j] = 0.0f;
    }
  }
}

float rothVerf(float* a, float* b, int n) {
  float sum = 0;
  int i, j;
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      sum += (float) fabs(a[i*n+j] - b[i*n+j]);
    }
  }
  return sum;
}

void print_mat(float* a, int n) {
  int i, j;
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      printf("%.1f\t", a[i*n+j]);
    }
    printf("\n");
  }
  printf("\n");
}


int main() {
  srand(100);
  int n = 4092;
  size_t totalsize = sizeof(float)*n*n;
  float *a, *b, *c;

  cudaMallocHost((void**) &a, totalsize);
  cudaMallocHost((void**) &b, totalsize);
  cudaMallocHost((void**) &c, totalsize);

  // construct first matrix
  make_identity(a, 0, 0, n, n);


  // second matrix
  make_identity(b, 0, 0, n, n);

  // allocate on device
  float *dev_a, *dev_b, *dev_c;
  cudaMalloc((void**) &dev_a, totalsize);
  cudaMalloc((void**) &dev_b, totalsize);
  cudaMalloc((void**) &dev_c, totalsize);

  // copy to device
  cudaMemcpy(dev_a, a, totalsize, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, totalsize, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, totalsize, cudaMemcpyHostToDevice);

  unsigned int grid_rows = n / BLOCK_SIZE;
  unsigned int grid_cols = n / BLOCK_SIZE;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  // intermediate matrix product
  MatrixMulKernel<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, n);
  cudaThreadSynchronize();

  // bring product back to cpu
  cudaMemcpy(c, dev_c, totalsize, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();

  // check a against the result d
  float *id;
  cudaMallocHost((void**) &id, totalsize);
  make_identity(id, 0, 0, n, n);
  printf("Sum: %f\n", rothVerf(c, id, n));
  
  // cleanup and exit
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  return 0;
}
