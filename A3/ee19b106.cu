#include <stdio.h>
#include <cuda.h>

using namespace std;

#define MAX 1024

__global__ void dkernel(int m, int n, int *t, int *p, int * ans){
    __shared__ unsigned int firstFree, start;
    __shared__ int pRevMap[MAX];
    __shared__ int endTimes[MAX];

    int core = threadIdx.x;

    if(core == 0){

        firstFree = MAX;
        start = 0;

        for(int i = 0; i < MAX; i++){
            pRevMap[i] = -1;
            endTimes[i] = 0;
        }
    }

    __syncthreads();

    for(int ptr = 0;ptr < n;ptr++){
        firstFree = MAX;
        __syncthreads();

        if(endTimes[core] <= start)
            atomicMin(&firstFree, core);

        __syncthreads();


        if(pRevMap[p[ptr]] == -1){
            if(core == firstFree){
                pRevMap[p[ptr]] = firstFree;

                endTimes[firstFree] = start + t[ptr];

                ans[ptr] = start + t[ptr];
            }
        }
        else{
            if(core == pRevMap[p[ptr]]){
                if(start < endTimes[core]){
                    start = endTimes[core];
                }
                endTimes[core] = start + t[ptr];
                ans[ptr] = start + t[ptr];
            }

        }
        __syncthreads();

    }


}

//Complete the following function
void operations ( int m, int n, int *executionTime, int *priority, int *result )  {
    int *d_executionTime, *d_priority, *d_result;

    cudaMalloc(&d_executionTime, n * sizeof(int));
    cudaMalloc(&d_priority, n * sizeof(int));
    cudaMalloc(&d_result, n * sizeof(int));

    cudaMemcpy(d_executionTime, executionTime, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_priority, priority, n * sizeof(int), cudaMemcpyHostToDevice);

    dkernel<<<1, m>>>(m, n, d_executionTime, d_priority, d_result);

    cudaMemcpy(result, d_result, n * sizeof(int), cudaMemcpyDeviceToHost);
}

int main(int argc,char **argv)
{
    int m,n;
    //Input file pointer declaration
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");
    
    //Checking if file ptr is NULL
    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &m );      //scaning for number of cores
    fscanf( inputfilepointer, "%d", &n );      //scaning for number of tasks
   
   //Taking execution time and priorities as input	
    int *executionTime = (int *) malloc ( n * sizeof (int) );
    int *priority = (int *) malloc ( n * sizeof (int) );
    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &executionTime[i] );
    }

    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &priority[i] );
    }

    //Allocate memory for final result output 
    int *result = (int *) malloc ( (n) * sizeof (int) );
    for ( int i=0; i<n; i++ )  {
        result[i] = 0;
    }
    
     cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    //==========================================================================================================
	

	operations ( m, n, executionTime, priority, result ); 
	
    //===========================================================================================================
    
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    
    // Output file pointer declaration
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    //Total time of each task: Final Result
    for ( int i=0; i<n; i++ )  {
        fprintf( outputfilepointer, "%d ", result[i]);
    }

    fclose( outputfilepointer );
    fclose( inputfilepointer );
    
    free(executionTime);
    free(priority);
    free(result);
    
    
    
}