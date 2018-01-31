
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define THREADS 4
#define SCHED static
const int N = 1000;
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
  int n_threads;
  #pragma omp parallel shared(n_threads) num_threads(THREADS)
  {
    #pragma omp single
      n_threads = omp_get_num_threads();
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
  #pragma omp parallel for num_threads(THREADS) schedule(SCHED)
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
  int n_threads;
  #pragma omp parallel shared(n_threads) num_threads(THREADS)
  {
    #pragma omp single
      n_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int start = tid*N/n_threads;
    int end = (tid+1)*N/n_threads;
    printf("\n\n");
    for (int i=start; i<end; i++) {
      printf("\n");
      for (int j=0; j<N; j++) {
	printf("%d\t", arr[i][j]);
	if (arr[i][j] > max) {
	  #pragma omp critical
	  max = arr[i][j];
      }
    }
  }
  }
  #pragma omp barrier
  return max;
}

int max_in_matrix_p2(int arr[N][N], int N) {
  int max = arr[0][0];
  #pragma omp parallel for num_threads(THREADS) schedule(SCHED)
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
  #pragma omp parallel for reduction(max:max) num_threads(THREADS) schedule(SCHED)
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      if (arr[i][j] > max)
	max = arr[i][j];
    }
  }
  return max;
}

void make_histogram(int hist[], int arr[N][N], int N, int M) {
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<10; k++) {
	if (k*M/10 <= arr[i][j] && arr[i][j] < (k+1)*M/10) {
	  hist[k]++;
	  break;
	}
      }
    }
  }
}

void make_histogram_p1(int hist[], int arr[N][N], int N, int M) {
  int n_threads;
  #pragma omp parallel shared(n_threads) num_threads(THREADS)
  {
    #pragma omp single
      n_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int start = tid*N/n_threads;
    int end = (tid+1)*N/n_threads;
    for (int i=start; i<end; i++) {
      for (int j=0; j<N; j++) {
	for (int k=0; k<10; k++) {
	  if (k*M/10 <= arr[i][j] && arr[i][j] < (k+1)*M/10) {
	    #pragma omp critical
	    hist[k]++;
	    break;
	  }
	}
      }
    }
  }
}

void make_histogram_p2(int hist[], int arr[N][N], int N, int M) {
  #pragma omp parallel for num_threads(THREADS) schedule(SCHED)
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<10; k++) {
	if (k*M/10 <= arr[i][j] && arr[i][j] < (k+1)*M/10) {
	  #pragma omp critical
	  hist[k]++;
	  break;
	}
      }
    }
  }
}

void make_histogram_p3(int hist[], int arr[N][N], int N, int M) {
  #pragma omp parallel for reduction(+:hist[:10]) num_threads(THREADS) schedule(SCHED)
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<10; k++) {
	if (k*M/10 <= arr[i][j] && arr[i][j] < (k+1)*M/10) {
	  hist[k]++;
	  break;
	}
      }
    }
  }
}

#include <sys/timeb.h>
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

// batch test: average of 10 runs
int main(void) {
  int (*A)[N] = malloc(sizeof(int[N][N]));
  int max;
  int hist[10];
  double sumtime = 0;
  double start;

  for (int i=0; i<10; i++) {
    start = read_timer();
    fill_matrix(A, N, M);
    max = max_in_matrix(A, N);
    make_histogram(hist, A, N, M);
    sumtime += read_timer() - start;
  }

  double avg_elapsed_ms = sumtime*100;
  printf("N: %d, M: %d, t (ms) = %f\n", N, M, avg_elapsed_ms);
  return 0;
}
