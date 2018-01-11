/* a simple openMP program */

#include <stdio.h>

#include <omp.h>

int main(int argc, char *argv[]){

  int num_threads = 9999;
  
  #pragma omp parallel // branch for # of threads
  {
    int thread_id = omp_get_thread_num();
    
    #pragma omp master // only run by master
    {
      num_threads = omp_get_num_threads();
    }

    #pragma omp single // only run by one thread
    {
      printf("Single!\n");
    }
    
    #pragma omp barrier
    
    printf("Hello from thread %d nthread %d\n",
	   thread_id, num_threads);
        
  } // End of Parallel region
  return 0;
}

