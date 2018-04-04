#include <stdio.h>
#include <stdlib.h>
#define N 10000

float getnum() {
  return rand()/((float) RAND_MAX);
}

void matrix_multiply(float* a, float* b, float* c, int n, int q) {
  int i, j, k;
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      float temp = 0.0f;
      for (k=0; k<n; k++) {
	temp += a[i*n+k]*b[k*n+j];
      }
      c[i*n+j] = temp;
    }
  }
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
      sum += abs(a[i*n+j] - b[i*n+j]);
    }
  }
  return sum;
}

void print_mat(float* a, int n) {
  int i, j;
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      printf("%.2f\t", a[i*n+j]);
    }
    printf("\n");
  }
  printf("\n");
}


int main() {
  srand(100);
  int n = N;
  int half = n>>1;
  float* x = malloc(sizeof(float)*(half)*(half));
  float* a = malloc(sizeof(float)*n*n);
  float* b = malloc(sizeof(float)*n*n);
  float* c = malloc(sizeof(float)*n*n);
  float* d = malloc(sizeof(float)*n*n);

  if ((x==NULL) || (a==NULL) || (b==NULL) || (c==NULL) ||
      (d==NULL)) {
    printf("Matrix allocation error\n");
    exit(1);
  }

  make_x(x, half);

  // construct first matrix
  make_identity(a, 0, 0, half, n);
  copy_x(a, x, 0, half, half, n);
  make_zero(a, half, 0, half, n);
  make_identity(a, half, half, half, n);

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

  // intermediate matrix product
  float* inter = malloc(sizeof(float)*n*n);
  //block_mult(a, b, inter, n, half);
  matrix_multiply(a, b, inter, n, half);
  free(b);
  
  // reuse old matrix
  //block_mult(inter, c, a, n, half);
  matrix_multiply(inter, c, a, n, half);
  
  // check a against the result d
  float sum = rothVerf(a, d, n);
  printf("Total Error: %f\n", sum);

  // cleanup and exit
  free(a);
  free(c);
  free(d);
  free(inter);
  return 0;
}
