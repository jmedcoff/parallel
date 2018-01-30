
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

const int N = 10;
const int M = 100;

void fill_matrix(int arr[N][N], int N, int M) {
  srand(1234);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      arr[i][j] = rand() % M;
    }
  }
}

void fill_matrix_p1(int arr[N][N], int N, int M) {
  srand(1234);
  int n_threads = omp_get_num_threads();
  #pragma omp parallel shared(arr)
  {
    int tid = omp_get_thread_num();
    int start = tid*N/n_threads;
    int end = (tid+1)*N/n_threads;
    int i, j;
    for (i=start; i<end; i++) {
      for (j=0; j<N; j++) {
	arr[i][j] = rand() % M;
      }
    }
  }
}

void fill_matrix_p2(int arr[N][N], int N, int M) {
  srand(1234);
  #pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      arr[i][j] = rand() % M;
    }
  }
}

int max_in_matrix(int arr[N][N], int N) {
  int max = arr[0][0];
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      if (arr[i][j] > max)
	max = arr[i][j];
    }
  }
  return max;
}

int max_in_matrix_p1(int arr[N][N], int N) {
  int max = arr[0][0];
  int n_threads = omp_get_num_threads();
  #pragma omp parallel shared(max)
  {
  int tid = omp_get_thread_num();
  int start = tid*N/n_threads;
  int end = (tid+1)*N/n_threads;
  for (int i=start; i<end; i++) {
    for (int j=0; j<N; j++) {
      if (arr[i][j] > max) {
	#pragma omp critical
	max = arr[i][j];
      }
    }
  }
  }
  return max;
}

int max_in_matrix_p2(int arr[N][N], int N) {
  int max = arr[0][0];
  #pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      if (arr[i][j] > max) {
	#pragma omp critical
	max = arr[i][j];
      }
    }
  }
  return max;
}

int max_in_matrix_p3(int arr[N][N], int N) {
  int max = arr[0][0];
  #pragma omp parallel for reduction(max:max)
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      if (arr[i][j] > max)
	max = arr[i][j];
    }
  }
  return max;
}

int main(void) {
  int (*A)[N] = malloc(sizeof(int[N][N]));
  fill_matrix(A, N, M);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
     printf("%d ", A[i][j]);
    }
    printf("\n");
  }
  int maximum = max_in_matrix_p3(A, N);
  printf("%d\n", maximum);
  free(A);
  return 0;
} // at this point, the random matrix
  // is being successfully created
  // and the maximum output as well.
