%%writefile obt.cu

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#define id(i, j, N) ((i) * N + (j))
#define ROW(p, N) ((p) / N)
#define COL(p, N) ((p) % N)
#define BOX(p, n) ((p) / (n * n * n) * n + ((p) % (n * n)) / n)

#define PUSH 0
#define POP 1
#define errck error_check(__LINE__, 2, d_boards, d_answer)

int* read_board(const char* file_name, int* N) {
    FILE* fp = fopen(file_name, "r");
    int* board = NULL;

    fscanf(fp, "%d", N);
    int total = *N * *N, i;
    board = (int*) calloc(total, sizeof(int));
    for (i = 0; i < total; ++ i)
        fscanf(fp, "%d", board + i);
    return board;
}


void error_check(int line_number, int arg_count, ...) {
    cudaError_t err=cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] %s at line %d\n", cudaGetErrorString(err), line_number);
        va_list ap;
        va_start(ap, arg_count);
        int i;
        for (i = 0; i < arg_count; ++ i) {
            int *arr = va_arg(ap, int*);
            if (arr != 0) cudaFree(arr);
        }
        va_end(ap);
        exit(-1);
    }
}

void print_board(int* board, int N, FILE* fp) {
    /*
        Prints the board to file 
    */

    int i, j;
    for (i = 0; i < N; ++ i) {
        for (j = 0; j < N; ++ j) {
            fprintf(fp, "%d ", board[id(i, j, N)]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
}

int check_partial_board(int* board, int n, int pos, int num) {
    /*
        Checks if num is valid at pos in the partially filled board
    */

    
    int N = n * n;
    int row_box = pos / (n * N), col_box = (pos % N) / n;

    int box_start_pos = row_box * n * N + col_box * n;

    
    //check row
    for (int j = ROW(pos, N) * N; j < (ROW(pos, N)+1) * N; ++ j)
        if (board[j] == num)
            return 0;

    // check col
    for (int j = COL(pos, N); j < N * N; j += N)
        if (board[j] == num)
            return 0;

    // check box
    for (int j = 0; j < N; ++ j)
        if (board[box_start_pos + (j / n) * N + (j % n)] == num)
            return 0;
    return 1;
}

__global__ void backtrack_kernel(int n, int *boards, int *found) {

    int N = n * n;
    int sz = N*N;

    
    extern __shared__ int shared[];
    int board_num = boards[blockIdx.x * sz + (threadIdx.x * N) + threadIdx.y];

    int *failed = shared + 4;
    int *stack = shared + N + 4;

    int top = 0;

    // all locations of empty tiles in stack
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int i;
        shared[0] = -1;
        for (i = 0; i < sz; ++ i) {
            if (boards[blockIdx.x * sz + i] == 0)
                stack[top ++] = i;
        }
        stack[top] = -1;
        top = 0;
    }
    
    int box_now = threadIdx.x / n * n + threadIdx.y / n;
    __syncthreads();

    int last_op = PUSH;    
    while (*found == -1) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            int stack_num = stack[top];
            shared[0] = stack_num % (sz); // record the first_empty_pos
            if (last_op == PUSH) {
                // first check if the board is filled
                if (stack_num == -1) {
                    // answer found
                    atomicCAS(found, -1, blockIdx.x);
                    shared[0] = -1;
                }
                // else initialize the number to try
                shared[1] = 1;
            } else {
                shared[1] = stack_num / (sz) + 1;
            }
        }
        if (threadIdx.y == 0) failed[threadIdx.x] = 0;
        __syncthreads();
        
        // find next valid number
        int first_empty_pos = shared[0], i = shared[1];
        if (first_empty_pos == -1) break;
        int num_to_try = i;
        if (ROW(first_empty_pos, N) == threadIdx.x || COL(first_empty_pos, N) == threadIdx.y || BOX(first_empty_pos, n) == box_now) {
            for (; i <= N; ++ i) if (i == board_num) failed[i - 1] = 1;
        }
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            for (i = num_to_try; i <= N && failed[i - 1] != 0; ++ i);
            shared[2] = first_empty_pos;   // record in shared memory 
            if (i <= N) {
                // push stack
                stack[top ++] = i * sz + first_empty_pos;
                shared[3] = i;
                last_op = PUSH;
            } else {
                // pop stack
                if (top == 0) shared[2] = -1;
                stack[top --] = first_empty_pos;
                shared[3] = 0;
                last_op = POP;
            }
        }
        __syncthreads();
        if (shared[2] == -1) break;
        
        if (threadIdx.x * N + threadIdx.y == shared[2])
            board_num = shared[3];

    }

    if (*found == blockIdx.x)
        boards[blockIdx.x * sz + (threadIdx.x * N) + threadIdx.y] = board_num;
}

int expand_on_cpu(int n, int *boards, int expand) {

    int N = n * n;
    int sz = N * N;

    int p1 = -1, p2 = 1;

    int num_empty_slots = expand - 1;
    int i;

    // initialize every slot as empty
    for (i = 1; i < expand; ++ i)
        boards[i * N * N] = -1;

    // expand on cpu
    while (num_empty_slots > 0 && num_empty_slots < expand) {
        
        p1 = (p1 + 1)%expand;
        int *curr_board = boards + p1 * sz;

        if (curr_board[0] == -1) continue;

        // first empty location in curr board
        int first_empty_pos;

        for (first_empty_pos = 0; first_empty_pos < N * N && curr_board[first_empty_pos] != 0; ++ first_empty_pos);

        if (first_empty_pos == N * N) {
            // answer found
            memcpy(boards, curr_board, sz * sizeof(int));
            return 1;
        }

        int num_and_pos_found = 0;
        for (i = 1; i <= N; ++ i)
            if (check_partial_board(curr_board, n, first_empty_pos, i)) {
                if (num_and_pos_found == 0)
                    num_and_pos_found = i * N * N + first_empty_pos;
                else {
                    // empty slots
                    if (num_empty_slots == 0) return 0;
                    int *board_new = boards + p2 * sz;
                    while (board_new[0] != -1) {
                        p2 = (p2 + 1)%expand;
                        board_new = boards + p2 * sz;
                    }
                    num_empty_slots --;
                    // modify board
                    memcpy(board_new, curr_board, sz * sizeof(int));
                    board_new[first_empty_pos] = i;
                }
            }
        if (num_and_pos_found == 0) {
            curr_board[0] = -1; // this board is empty
            num_empty_slots ++;
        } else {
            // modify current board
            curr_board[num_and_pos_found % (N * N)] = num_and_pos_found / (N * N);
        }
            
    }
    return 0;
}


int solve(int* board, int n, int expand, int tile_size, FILE* fp) {
    int N = n * n;

    int *boards = (int*) malloc(N * N * expand * sizeof(int));
    memcpy(boards, board, N * N * sizeof(int));
    
    int answer = expand_on_cpu(n, boards, expand);
    
    if (answer == 1) {
        print_board(boards, N, fp);
    } else if (answer == 0) {
        printf("CPU expansion finished.\n");

        // malloc arrays for cuda
        int *d_boards = 0, *d_answer = 0;

        cudaMalloc((void**) &d_boards, N * N * expand * sizeof(int)); errck;
        cudaMemcpy(d_boards, boards, N * N * expand * sizeof(int), cudaMemcpyHostToDevice); errck;
        cudaMalloc((void**) &d_answer, sizeof(int)); errck;
        cudaMemset(d_answer, -1, sizeof(int)); errck;

        dim3 grid_dim(expand);
        //dim3 block_dim(N, N);
        dim3 block_dim(N, N);
        backtrack_kernel<<<grid_dim, block_dim, (N * N + N + 4) * sizeof(int)>>>(n, d_boards, d_answer); errck;

        cudaMemcpy(&answer, d_answer, sizeof(int), cudaMemcpyDeviceToHost); errck;
        printf("GPU search finished. %d\n", answer);
        
        if (answer >= 0) {
            cudaMemcpy(boards, d_boards + answer * N * N, N * N * sizeof(int), cudaMemcpyDeviceToHost); errck;
            answer = 1;
            print_board(boards, N, fp);
        }

        cudaFree(d_boards);
        cudaFree(d_answer);
    }

    free(boards);
    return answer;

}

int main(int argc, char* argv[]) {

    if (argc < 5) {
        printf("Usage:\n");
        printf("./sudoku_bt <input> <output> <expand> <block_size>\n");
        exit(-1);
    }

    int N;
    int* board = read_board(argv[1], &N);
    int n = sqrt((double)N);
    printf("Started Solving...\n");

    FILE* fp = fopen(argv[2], "w");
    int expand = (int) atoi(argv[3]);
    int tile_size = (int) atoi(argv[4]);
    int ans = solve(board, n, expand, tile_size, fp);

    if (ans == 1) printf("Answer found. \n");
    else printf("Answer not found :(\n");

    free(board);
    return 0;
}









