#include <stdio.h>
#include <stdlib.h>

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

struct blockmat {
  float* block1;
  float* block2;
  float* block3;
  float* block4;
};

void block_mult(float* a, float* b, float* c, int n, int bsize) {
  // assuming n is a multiple of bsize
  // NOT WORKING AT THE MOMENT
  int i, j, k, kprime, jprime;
  float sum;
  int en = bsize*(n/bsize);

  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      c[i*n+j] = 0.0f;
    }
  }

  for (kprime=0; kprime<en; kprime+=bsize) {
    for (jprime=0; jprime<en; jprime+=bsize) {
      for (i=0; i<n; i++) {
	for (j=jprime; j<jprime+bsize; j++) {
	  sum = c[i*n+j];
	  for (k=kprime; k<kprime+bsize; k++) {
	    sum += a[i*n+k] + b[k*n+j];
	  }
	  c[i*n+j] = sum;
	}
      }
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
  int n = 1024;
  int half = n>>1;
  float* x = malloc(sizeof(float)*(half)*(half));
  float* a = malloc(sizeof(float)*n*n);
  float* b = malloc(sizeof(float)*n*n);
  float* c = malloc(sizeof(float)*n*n);
  float* d = malloc(sizeof(float)*n*n);

  if ((x==NULL) || (a==NULL) || (b==NULL) || (c==NULL) ||
      (d==NULL)) {
    printf("Oh shit fam, memory machine broke\n");
  }
  else
    printf("memory machine work\n");

  make_x(x, half);

  // construct first matrix
  make_identity(a, 0, 0, half, n);
  printf("asdf\n");  
  copy_x(a, x, 0, half, half, n);

  make_zero(a, half, 0, half, n);
  make_identity(a, half, half, half, n);
  print_mat(a, n);
  
  // second matrix
  make_identity(b, 0, 0, half, n);
  copy_2x(b, x, 0, half, half, n);
  make_zero(b, half, 0, half, n);
  make_negidentity(b, half, half, half, n);
  print_mat(b, n);

  // third
  make_identity(c, 0, 0, half, n);
  copy_negx(c, x, 0, half, half, n);
  make_zero(c, half, 0, half, n);
  make_identity(c, half, half, half, n);
  print_mat(c, n);

  // result
  make_result(d, n);
  print_mat(d, n);

  // intermediate matrix product
  float* inter = malloc(sizeof(float)*n*n);
  //block_mult(a, b, inter, n, half);
  matrix_multiply(a, b, inter, n, half);
  // reuse old matrix
  //block_mult(inter, c, a, n, half);
  matrix_multiply(inter, c, a, n, half);
  print_mat(a, n);
  // check a against the result d
  float sum = rothVerf(a, d, n);
  printf("Total Error: %f\n", sum);
  free(a);
  free(b);
  free(c);
  free(d);
  free(inter);
  return 0;
}
