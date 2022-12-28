#include<iostream>
#include<sys/time.h>
#include<cuda.h>
// DEBUG = 1 to see print output
#define DEBUG 0
#define print(fmt, ...) do { if (DEBUG) fprintf(stderr, fmt, __VA_ARGS__); } while (0)
#define BLOCK_DIM 32
#define TILE_DIM 32


using namespace std;

// function to print matrix for debugging
void printMatrix(int rows, int cols, int *matrix){
		if(DEBUG){
			for(long i = 0; i < rows; i++){
					for(long j = 0; j < cols; j++){
							print("%d ", matrix[i*cols+j]);
					}
					print("%s \n", "");
			}
		}
}

// write kernels here...
__global__  static void matMul(int *left, int *right, int *output, int p, int q, int r)
{
		
    /***
    Multiplies matrices left and right of dimension p * q and q * r respectively.
    Stores result in output matrix of size p * r
    **/
		
	long accu = 0;
 
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ long leftTile[TILE_DIM][TILE_DIM];
    __shared__ long rightTile[TILE_DIM][TILE_DIM];


    for (int k = 0; k < (TILE_DIM + q - 1)/TILE_DIM; k++) {
				 // k is the tile num

         if (k * TILE_DIM + threadIdx.x < q && row < p)
             leftTile[threadIdx.y][threadIdx.x] = left[row * q + k * TILE_DIM + threadIdx.x];
         else
             leftTile[threadIdx.y][threadIdx.x] = 0.0;

         if (k * TILE_DIM + threadIdx.y < q && col < r)
             rightTile[threadIdx.y][threadIdx.x] = right[(k * TILE_DIM + threadIdx.y) * r + col];
         else
             rightTile[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
             accu += leftTile[threadIdx.y][n] * rightTile[n][threadIdx.x];

         __syncthreads();
    }
	
		// if pos is valid in output array
    if (row < p && col < r)
        output[((blockIdx.y * blockDim.y + threadIdx.y) * r) + (blockIdx.x * blockDim.x)+ threadIdx.x] = accu;
}



__global__ void transpose(int *odata, int *idata, int width, int height)
{
	__shared__ int block[TILE_DIM][TILE_DIM+1];
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

	if((xIndex < width) && (yIndex < height))
	{
		int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}
	__syncthreads();

	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

	if((xIndex < height) && (yIndex < width))
	{
		int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}

}


__global__ void add(int *A, int *B, int width, int height)
{
   int xIndex = blockDim.x * blockIdx.y + threadIdx.y;
   int yIndex = blockDim.y * blockIdx.x + threadIdx.x;
   	
	if((xIndex < height) && (yIndex < width))
	{
		int index = yIndex * height + xIndex;
		A[index] += B[index];
	}
}

// function to compute the output matrix
void compute(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, int *h_matrixC, int *h_matrixD, int *h_matrixX) {

	//B transpose
	int *h_matrixB_T, *d_matrixB, *d_matrixB_T;

	h_matrixB_T = (int*) malloc(p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * p * sizeof(int));
	cudaMalloc(&d_matrixB_T, p * q * sizeof(int));

	cudaMemcpy(d_matrixB, h_matrixB, q * p * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB_T, h_matrixB_T, p * q * sizeof(int), cudaMemcpyHostToDevice);

	dim3 grid(ceil(float(p) / BLOCK_DIM), ceil(float(q) / BLOCK_DIM), 1);
    dim3 block(BLOCK_DIM, BLOCK_DIM, 1);

	transpose<<< grid, block >>>(d_matrixB_T, d_matrixB, p, q);
	cudaDeviceSynchronize();

	cudaMemcpy(h_matrixB_T, d_matrixB_T, p * q * sizeof(int), cudaMemcpyDeviceToHost);

	print("%s \n", "Matrix B before transpose");
	printMatrix(q, p, h_matrixB);
    print("%s \n", "Matrix B after transpose");
	printMatrix(p, q, h_matrixB_T);
  
    print("%s \n", "********************************************************");

    //D transpose
	int *h_matrixD_T, *d_matrixD, *d_matrixD_T;

	h_matrixD_T = (int*) malloc(r * s * sizeof(int));
	cudaMalloc(&d_matrixD, s * r * sizeof(int));
	cudaMalloc(&d_matrixD_T, r * s * sizeof(int));

	cudaMemcpy(d_matrixD, h_matrixD, s * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD_T, h_matrixD_T, r * s * sizeof(int), cudaMemcpyHostToDevice);

	dim3 grid1(ceil(float(r) / BLOCK_DIM), ceil(float(s) / BLOCK_DIM), 1);
    dim3 block1(BLOCK_DIM, BLOCK_DIM, 1);

	transpose<<< grid1, block1 >>>(d_matrixD_T, d_matrixD, r, s);
	cudaDeviceSynchronize();

	cudaMemcpy(h_matrixD_T, d_matrixD_T, r * s * sizeof(int), cudaMemcpyDeviceToHost);

	print("%s \n", "Matrix D before transpose");
	printMatrix(s, r, h_matrixD);
	print("%s \n", "Matrix D after transpose");
	printMatrix(r, s, h_matrixD_T);

    print("%s \n", "********************************************************");

    // A <= A + B_T

	int *d_matrixA;

	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);

	dim3 grid2(ceil(float(p) / BLOCK_DIM), ceil(float(q) / BLOCK_DIM), 1);
    dim3 block2(BLOCK_DIM, BLOCK_DIM, 1);

	print("%s \n", "Before Addition");
    print("%s \n", "A");
	printMatrix(p, q, h_matrixA);
    print("%s \n", "B");
	printMatrix(p, q, h_matrixB_T);
	
	add<<< grid2, block2 >>>(d_matrixA, d_matrixB_T, p, q);
	cudaDeviceSynchronize();
	cudaMemcpy(h_matrixA, d_matrixA, p * q * sizeof(int), cudaMemcpyDeviceToHost);
  
	print("%s \n", "Matrix A After Addition");
	printMatrix(p, q, h_matrixA);

    print("%s \n", "********************************************************");

    // E = C * D_T
	int *h_matrixE, *d_matrixE, *d_matrixC;

	h_matrixE = (int*) malloc(q * s * sizeof(int));
	cudaMalloc(&d_matrixE, q * s * sizeof(int));
	cudaMalloc(&d_matrixC, q * r * sizeof(int));

	cudaMemcpy(d_matrixC, h_matrixC, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD_T, h_matrixD_T, r * s * sizeof(int), cudaMemcpyHostToDevice);

	dim3 grid3(ceil(float(s) / BLOCK_DIM), ceil(float(q) / BLOCK_DIM), 1);
    dim3 block3(BLOCK_DIM, BLOCK_DIM, 1);

	print("%s \n", "Before multiplication");
    print("%s \n", "C");
	printMatrix(q, r, h_matrixC);
    print("%s \n", "D_T");
	printMatrix(r, s, h_matrixD_T);

	matMul<<< grid3, block3 >>>(d_matrixC, d_matrixD_T, d_matrixE, q, r, s);
	cudaDeviceSynchronize();

	cudaMemcpy(h_matrixE, d_matrixE, q * s * sizeof(int), cudaMemcpyDeviceToHost);
	print("%s \n", "E = C * D_T");
	printMatrix(q, s, h_matrixE);


    print("%s \n", "********************************************************");

    //X = A * E;

	int *d_matrixX;

	cudaMalloc(&d_matrixX, p * s * sizeof(int));
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixE, h_matrixE, q * s * sizeof(int), cudaMemcpyHostToDevice);


	dim3 grid4(ceil(float(s) / BLOCK_DIM), ceil(float(p) / BLOCK_DIM), 1);
    dim3 block4(BLOCK_DIM, BLOCK_DIM, 1);

	print("%s \n", "Before multiplication");
    print("%s \n", "A");
	printMatrix(p, q, h_matrixA);
    print("%s \n", "E");
	printMatrix(q, s, h_matrixE);

	matMul<<< grid4, block4 >>>(d_matrixA, d_matrixE, d_matrixX, p, q, s);
	cudaDeviceSynchronize();

	cudaMemcpy(h_matrixX, d_matrixX, p * s * sizeof(int), cudaMemcpyDeviceToHost);

	print("%s \n", "X = A * E");
	printMatrix(p, s, h_matrixX);

	print("%s \n", "********************************************************");

	// deallocate the memory
	free(h_matrixB_T);
	cudaFree(d_matrixB);
	cudaFree(d_matrixB_T);

	free(h_matrixD_T);
	cudaFree(d_matrixD);
	cudaFree(d_matrixD_T);

	cudaFree(d_matrixA);

	free(h_matrixE);
	cudaFree(d_matrixE);
	cudaFree(d_matrixC);

	cudaFree(d_matrixX);






	

}


// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}



// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r, s;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixX;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d %d", &p, &q, &r, &s);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * p * sizeof(int));
	matrixC = (int*) malloc(q * r * sizeof(int));
	matrixD = (int*) malloc(s * r * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, p);
	readMatrix(inputFilePtr, matrixC, q, r);
	readMatrix(inputFilePtr, matrixD, s, r);

	// allocate memory for output matrix
	matrixX = (int*) malloc(p * s * sizeof(int));

	// call compute function to get the output matrix. it is expected that 
	// the compute function will store the result in matrixX.
	gettimeofday(&t1, NULL);
	compute(p, q, r, s, matrixA, matrixB, matrixC, matrixD, matrixX);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixX, p, s);

	// close files
    fclose(inputFilePtr);
    fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixX);

	return 0;
}