#include <stdio.h>
#include <cuda.h>
#include <thrust/sort.h>

#define MAX_TRAINS 100032
#define MAX_CLASSES 32
#define MAX_REQS 5000
#define BLOCK_DIM 1024
#define MAX_DIST 50

typedef int trainInfoArr[3];
typedef int trainClassInfoArr[MAX_CLASSES];
typedef int requestsDataArr[5];
typedef int trainClassOccInfoArr[MAX_CLASSES][MAX_DIST];



__global__ void dkernel(trainInfoArr *trainInfo, trainClassInfoArr *trainClassInfo, trainClassOccInfoArr *trainClassOccInfo,requestsDataArr *reqData, int *reqId, int *reqMap, int *reqRes, unsigned int *numPass, unsigned int *numFail, unsigned int *seatsBooked, int n, int nT){
    int trainNum, classNum;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = threadIdx.x;

    trainNum = row;
    classNum = col;

    int src, dst, dirn;
    src = trainInfo[trainNum][1];
    dst = trainInfo[trainNum][2];

    if(src >= dst){
        dirn = 1;
    }
    else{
        dirn = -1;
    }
    
    int cap;
    cap = trainClassInfo[trainNum][classNum];


    for(int ptr = 0; ptr < n; ptr++){
        int tr, cl;
        int currReq = reqId[ptr];
        tr = reqData[currReq][0];
        cl = reqData[currReq][1];
    
        if(tr == trainNum && cl == classNum){
            //process this request as it belongs to this train, class
            int reqPos = ptr;
            int rId = reqId[reqPos];
            int rIdx = reqMap[reqPos];
            int rSrc = reqData[rIdx][2];
            int rDst = reqData[rIdx][3];
            int numSeats = reqData[rIdx][4];
            bool isPoss = true;

            if(dirn == -1){
                
                //first check if it is possible to process this request

                if(rSrc < src || rDst > dst){
                    //not possible
                    isPoss = false;
                }
                else{
                    for(int k = rSrc; k < rDst; k++){
                        //check if this spot is empty
                        if(trainClassOccInfo[tr][cl][k-src] + numSeats > cap){
                            //not possible
                            isPoss = false;
                            break;
                        }

                    }
                }
  
                //update values accordingly

                if(isPoss){
                    for(int k = rSrc; k < rDst; k++){
                        trainClassOccInfo[tr][cl][k-src] += numSeats;

                    }
                    reqRes[rIdx] = 1;
                    atomicInc(numPass, MAX_REQS);
                    atomicAdd(seatsBooked,  numSeats * (rDst- rSrc));
                }
                else{
                    reqRes[rIdx] = 0;
                    atomicInc(numFail, MAX_REQS);
                }

            }
            else{
                if(rSrc > src || rDst < dst){
                    //not possible
                    isPoss = false;
                }
                else{
                    for(int k = rSrc; k > rDst; k--){
                        //if this spot is empty
                        if(trainClassOccInfo[tr][cl][k-dst] + numSeats > cap){
                            //not possible
                            isPoss = false;
                            break;
                        }

                    }
                }

                if(isPoss){
                    for(int k = rSrc; k > rDst; k--){
                        trainClassOccInfo[tr][cl][k-dst] += numSeats;

                    }
                    reqRes[rIdx] = 1;
                    atomicInc(numPass, MAX_REQS);
                    atomicAdd(seatsBooked, numSeats * (rSrc - rDst));
                }
                else{
                    reqRes[rIdx] = 0;
                    atomicInc(numFail, MAX_REQS);
                }
            }
        }
    
    }
}

int main(void)
{
  
        int numTrains;
        scanf("%d", &numTrains);
        trainInfoArr *h_trainInfo, *d_trainInfo;
        trainClassInfoArr *h_trainClassInfo, *d_trainClassInfo;
        trainClassOccInfoArr *h_trainClassOccInfo, *d_trainClassOccInfo;

        size_t trainInfoSize = MAX_TRAINS*3*sizeof(int);
        h_trainInfo = (trainInfoArr *)malloc(trainInfoSize);
        memset(h_trainInfo, 0, trainInfoSize);

        size_t trainClassOccInfoSize = MAX_TRAINS * 32 * 50 * sizeof(int);
        h_trainClassOccInfo = (trainClassOccInfoArr *)malloc(trainClassOccInfoSize);
        memset(h_trainClassOccInfo, 0, trainClassOccInfoSize);
 
        
        size_t trainClassInfoSize = MAX_TRAINS*32*sizeof(int);
        h_trainClassInfo = (trainClassInfoArr *)malloc(trainClassInfoSize);
        memset(h_trainClassInfo, 0, trainClassInfoSize);

        for(int i = 0; i < numTrains; i++){
            int trainNum, numClasses, src, dst;
            scanf("%d", &trainNum);
            scanf("%d", &numClasses);
            scanf("%d", &src);
            scanf("%d", &dst);
            h_trainInfo[trainNum][0] = numClasses;
            h_trainInfo[trainNum][1] = src;
            h_trainInfo[trainNum][2] = dst;  
            for(int j = 0; j < numClasses; j++){
                int classNum, cap;
                scanf("%d", &classNum);
                scanf("%d", &cap);

                h_trainClassInfo[trainNum][classNum] = cap;

            }
        }

        cudaMalloc(&d_trainInfo, trainInfoSize);
        cudaMalloc(&d_trainClassInfo, trainClassInfoSize);
        cudaMalloc(&d_trainClassOccInfo, trainClassOccInfoSize);
        // Do memcopies to GPU
        cudaMemcpy(d_trainInfo, h_trainInfo, trainInfoSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_trainClassInfo, h_trainClassInfo, trainClassInfoSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_trainClassOccInfo, h_trainClassOccInfo, trainClassOccInfoSize, cudaMemcpyHostToDevice);

        int numBatches;
        scanf("%d", &numBatches);

        for(int i = 0; i < numBatches; i++){
            int numRequests;
            scanf("%d", &numRequests);

            requestsDataArr *h_requestsData, *d_requestsData;
            int *h_requestId, *d_requestId, *h_requestRes, *d_requestRes, *h_requestMap, *d_requestMap;
            unsigned int *h_numPass, *h_numFail, *d_numPass, *d_numFail, *h_seatsBooked, *d_seatsBooked;

            h_numPass = (unsigned int *)malloc(sizeof(int));
            h_numFail = (unsigned int *)malloc(sizeof(int));
            h_seatsBooked = (unsigned int *)malloc(sizeof(int));
            
            cudaMalloc(&d_numPass, sizeof(unsigned int));
            cudaMalloc(&d_numFail, sizeof(unsigned int));
            cudaMalloc(&d_seatsBooked, sizeof(unsigned int));

            cudaMemset(d_numPass, 0, sizeof(int));
            cudaMemset(d_numFail, 0, sizeof(int));
            cudaMemset(d_seatsBooked, 0, sizeof(int));

            size_t requestsDataSize = MAX_REQS*5*sizeof(int);
            h_requestsData = (requestsDataArr *)malloc(requestsDataSize);
            memset(h_requestsData, 0, requestsDataSize);

            h_requestId = (int *)malloc(MAX_REQS*sizeof(int));
            memset(h_requestId, 0, MAX_REQS*sizeof(int));

            h_requestMap = (int *)malloc(MAX_REQS*sizeof(int));
            memset(h_requestMap, 0, MAX_REQS*sizeof(int));

            h_requestRes = (int *)malloc(MAX_REQS*sizeof(int));

            for(int j = 0; j< numRequests; j++){
                int rId, rTrainNum, rClassNum, rSrc, rDst, rNumSeats;
                scanf("%d", &rId);
                scanf("%d", &rTrainNum);
                scanf("%d", &rClassNum);
                scanf("%d", &rSrc);
                scanf("%d", &rDst);
                scanf("%d", &rNumSeats);
                h_requestId[j] = rId;
                h_requestMap[j] = j;
                h_requestsData[j][0] = rTrainNum;
                h_requestsData[j][1] = rClassNum;
                h_requestsData[j][2] = rSrc;
                h_requestsData[j][3] = rDst;
                h_requestsData[j][4] = rNumSeats;
            }

          
            //thrust::sort(h_requestId, h_requestId + numRequests);
            thrust::sort_by_key(h_requestId, h_requestId + numRequests, h_requestMap);

            cudaMalloc(&d_requestsData, requestsDataSize);
            cudaMalloc(&d_requestId, MAX_REQS*sizeof(int));
            cudaMalloc(&d_requestMap, MAX_REQS*sizeof(int));
            cudaMalloc(&d_requestRes, MAX_REQS*sizeof(int));
            
            cudaMemcpy(d_requestsData, h_requestsData, requestsDataSize, cudaMemcpyHostToDevice);
            cudaMemcpy(d_requestId, h_requestId, MAX_REQS*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_requestMap, h_requestMap, MAX_REQS*sizeof(int), cudaMemcpyHostToDevice);
            
            dim3 grid(1, ceil(float(MAX_TRAINS * MAX_CLASSES) / BLOCK_DIM), 1);
            dim3 block(MAX_CLASSES, BLOCK_DIM/MAX_CLASSES, 1);

      

    
            dkernel<<< grid, block >>>(d_trainInfo, d_trainClassInfo, d_trainClassOccInfo,d_requestsData, d_requestId, d_requestMap, d_requestRes, d_numPass, d_numFail, d_seatsBooked, numRequests, numTrains);

            cudaMemcpy(h_requestRes, d_requestRes, MAX_REQS*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_numPass, d_numPass, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_numFail, d_numFail, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_seatsBooked, d_seatsBooked, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            for(int j = 0; j < numRequests; j++){
                if(h_requestRes[j]){
                    printf("success\n");
                }
                else{
                    printf("failure\n");
                }
            }

            printf("%u %u\n", *h_numPass, *h_numFail);
            printf("%u\n", *h_seatsBooked);

            //free
            free(h_numPass);
            free(h_numFail);
            free(h_seatsBooked);

            cudaFree(d_numPass);
            cudaFree(d_numFail);
            cudaFree(d_seatsBooked);
            
            free(h_requestsData);
            free(h_requestId);
            free(h_requestMap);
            free(h_requestRes);

            cudaFree(d_requestsData);
            cudaFree(d_requestId);
            cudaFree(d_requestMap);
            cudaFree(d_requestRes);
            


            
        }
    return 0;
}