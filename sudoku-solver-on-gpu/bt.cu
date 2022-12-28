%%writefile bt.cu

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


__device__ int check_partial_board_d(int* board, int n, int pos, int num) {
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

    int N = n * n, i;
    int sz = N*N;

    int local_board_id = threadIdx.x;
    int global_board_id = blockIdx.x * blockDim.x + local_board_id;

    
    // use shared memory
    extern __shared__ int shared[];
    int start_pos = local_board_id * sz;

    for (i = 0; i < sz; ++ i)
        shared[i + start_pos] = boards[global_board_id * sz + i];

    int *board = shared + start_pos;
    int *stack = shared + blockDim.x * sz + start_pos;

    int last_op = PUSH;    
    int top = 0, first_empty_pos = 0;

    while (*found == -1) {
        int num_to_try;
        if (last_op == PUSH) {
            
            for (;first_empty_pos < sz && board[first_empty_pos] != 0; ++ first_empty_pos);

            
            if (first_empty_pos == sz) {
                // answer found
                int old = atomicCAS(found, -1, global_board_id);
                if (old == -1) {
                    // copy back to global memory
                    for (i = 0; i < sz; ++ i)
                        boards[global_board_id * sz + i] = board[i];
                }
                break;
            }
            num_to_try = 1;
        } 
        else {
            // read stack top and restore
            int stack_num = stack[top];
            first_empty_pos = stack_num % (sz);
            num_to_try = board[first_empty_pos] + 1;
        }

        // next valid number
        for (;num_to_try <= N; ++ num_to_try)
            if (check_partial_board_d(board, n, first_empty_pos, num_to_try)) {
                // push 
                stack[top ++] = first_empty_pos;
                board[first_empty_pos] = num_to_try;

                last_op = PUSH;
                break;
            }

        // no valid number found, backtrack
        if (num_to_try > N) {
            if (top == 0) break;
            board[first_empty_pos] = 0;
            top --;
            last_op = POP;
        }
    }
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

        dim3 grid_dim(expand / tile_size);
        //dim3 block_dim(N, N);
        dim3 block_dim(tile_size);
        backtrack_kernel<<<grid_dim, block_dim, 2 * tile_size * N * N * sizeof(int)>>>(n, d_boards, d_answer); errck;

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









