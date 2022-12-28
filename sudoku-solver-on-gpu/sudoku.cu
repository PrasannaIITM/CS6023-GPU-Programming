%%writefile sudoku.cu

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#define id(i, j, N) ((i) * N + (j))
#define ROW(p, N) ((p) / N)
#define COL(p, N) ((p) % N)
#define BOX(p, n) ((p) / (n * n * n) * n + ((p) % (n * n)) / n)


#define BACKTRACK 1

#define BOARD_DIM 9

#define GROUP_DIM 3

#define BATCH_SIZE 15000

#define THREADS_PER_BLOCK 81

typedef struct board {
  uint16_t cells[BOARD_DIM * BOARD_DIM];
} board_t;

size_t time_ms() {
  struct timeval tv;
  if(gettimeofday(&tv, NULL) == -1) {
    perror("gettimeofday");
    exit(2);
  }
  
  // Convert timeval values to milliseconds
  return tv.tv_sec*1000 + tv.tv_usec/1000;
}

__host__ __device__ uint16_t digit_to_cell(int digit) {
  /*
    encode input digit in a mask
    input digit 0 is treated specially, it indicates a blank cell
    any value from one to nine is possible, so bits from 1 to 9 are set
  */

  if(digit == 0) {
    // blank cell, numbers 1-9 are possible.
    return 0x3FE;
  } else {
    // set the corresponding bit.
    return 1<<digit;
  }
}

__host__ __device__ int cell_to_digit(uint16_t cell) {
  
  /*
    decode the cell, i.e.
    get the index of the least-significant bit in this cell's value
    if more than one bits are set, returns 0
  */

#if defined(__CUDA_ARCH__)
  int msb = __clz(cell);
  int lsb = sizeof(unsigned int)*8 - msb - 1;
#else
  int lsb = __builtin_ctz(cell);
#endif

  if(cell == 1<<lsb) return lsb;
  else return 0;
}

__global__ void solve_boards_kernel(board_t* original_board)
{

  //row, col and box of the current cell
  int row = threadIdx.x/BOARD_DIM;
  int col = threadIdx.x%BOARD_DIM;
  int box = (col/GROUP_DIM) + GROUP_DIM*(row/GROUP_DIM);

  //top left of each box(make dynamic for N)
  int box_start_pos[9] = {0,3,6,27,30,33,54,57,60};

  //load the correct board in shared memory
  __shared__ board_t sh_board; 
  sh_board.cells[threadIdx.x] = original_board[blockIdx.x].cells[threadIdx.x];
 
  __syncthreads();

  //continue to constraint propagate flag
  int remaining = 1;

  //continue till at least one flag is non zero in current board
  while(__syncthreads_count(remaining) > 0)
    {

      remaining = 0;  
      
      uint16_t cell =  sh_board.cells[threadIdx.x];

      if(cell_to_digit(cell) == 0)
        {
          //Look through all the peers of this cell to propogate constraints

          //rows
          int r = row*BOARD_DIM;
          
          int i = 0;
          while(i < BOARD_DIM)
            {
              //update possible values of current according to the constraints
             
              int val = cell_to_digit(sh_board.cells[r+i]);
              if(val != 0)
                {
                  uint16_t tmp = cell;
                  //remove val from possible values of this cell
                  cell &= ~(1<<val);
                  if(tmp != cell)
                      remaining = 1;
                }
              i++;
            }

          //cols
          int j = 0;
          while(j < BOARD_DIM)
            {
              //update possible values of current according to the constraints
              int val = cell_to_digit(sh_board.cells[col+j*9]);
              if(val != 0)
                {
                  uint16_t tmp = cell;
                  //remove val from possible values of this cell
                  cell &= ~(1<<val);
                  if(tmp != cell)
                      remaining = 1;
                }
              j++;
            }


          //box
          uint16_t start = box_start_pos[box];
          for(i = 0; i < GROUP_DIM; i++)
            {
              for(j = 0; j < GROUP_DIM; j++)
                {
                  //update possible values of current according to the constraints
                  int val = cell_to_digit(sh_board.cells[start + i * BOARD_DIM + j]);
                  if(val != 0)
                    {
                      uint16_t tmp = cell;
                      //remove val from possible values of this cell
                      cell &= ~(1<<val);
                      if(tmp != cell)
                          remaining = 1;
                    }
                }
            }
          //update cells value in shared memory 
          sh_board.cells[threadIdx.x] = cell;
          
        }
      
    }
 
  //update original board from shared memory
  original_board[blockIdx.x].cells[threadIdx.x] = sh_board.cells[threadIdx.x];
        
}

void solve_boards(board_t* boards, size_t num_boards)
{
  
  board_t* d_boards;

  cudaMalloc(&d_boards, num_boards*sizeof(board_t));
  
  cudaMemcpy(d_boards, boards,  num_boards*sizeof(board_t), cudaMemcpyHostToDevice);
  
  solve_boards_kernel<<<num_boards,THREADS_PER_BLOCK>>>(d_boards);
  cudaDeviceSynchronize();
  
  cudaMemcpy(boards, d_boards,  num_boards*sizeof(board_t), cudaMemcpyDeviceToHost);

}

// check if num at position p is valid given a partially filled board
int check_partial_board(board_t* board, int n, int p, int num) {
    int j;
    int N = n * n;
    int box_row = p / (n * N);
    int box_col = (p % N) / n;
    int box_top_left = box_row * n * N + box_col * n;
    int now_row = ROW(p, N);
    for (j = now_row * N; j < (now_row + 1) * N; ++ j)
        if (cell_to_digit(board->cells[j]) == num)
            return 0;
    // check col
    for (j = COL(p, N); j < N * N; j += N)
        if (cell_to_digit(board->cells[j]) == num)
            return 0;
    // check box
    for (j = 0; j < N; ++ j)
        if (cell_to_digit(board->cells[box_top_left + (j / n) * N + (j % n)]) == num)
            return 0;
    return 1;
}

int backtracking_dfs(board_t* board, int n, int p) {
    int N = n * n;
    int i;
    if (p == N * N) {
        return 1;
    } else {
        uint16_t tmp = board->cells[p];
        if (cell_to_digit(board->cells[p]) == 0) {
            for (i = 0; i < N; ++ i) {
                if(tmp & (1<<(i+1)))
                {
                  if (check_partial_board(board, n, p, i + 1)) {
                      board->cells[p] = digit_to_cell(i + 1);
                      int ret = backtracking_dfs(board, n, p + 1);
                      if (ret) return ret;
                  }
                }
            }
            board->cells[p] = tmp;
        } else {
            return backtracking_dfs(board, n, p + 1);
        }
        return 0;
    }
}

int backtracking(board_t* board, int n) {
    int ans_num = backtracking_dfs(board, n, 0);
    return ans_num;
}



bool read_board(board_t* output, const char* str) {
  for(int index=0; index<BOARD_DIM*BOARD_DIM; index++) {
    if(str[index] < '0' || str[index] > '9') return false;

    // Convert the character value to an equivalent integer
    int value = str[index] - '0';

    // Set the value in the board
    output->cells[index] = digit_to_cell(value);
  }

  return true;
}

void print_board(board_t* board) {
  for(int row=0; row<BOARD_DIM; row++) {
    // Print horizontal dividers
    if(row != 0 && row % GROUP_DIM == 0) {
      for(int col=0; col<BOARD_DIM*2+BOARD_DIM/GROUP_DIM; col++) {
        printf("-");
      }
      printf("\n");
    }

    for(int col=0; col<BOARD_DIM; col++) {
      // Print vertical dividers
      if(col != 0 && col % GROUP_DIM == 0) printf("| ");

      // Compute the index of this cell in the board array
      int index = col + row * BOARD_DIM;

      // Get the index of the least-significant bit in this cell's value
      int digit = cell_to_digit(board->cells[index]);

      // Print the digit if it's not a zero. Otherwise print a blank.
      if(digit != 0) printf("%d ", digit);
      else if(digit == 0) printf("%d ", board->cells[index]);
      else printf("  ");
    }
    printf("\n");
  }
  printf("\n");
}

void check_solutions(board_t* boards, board_t* solutions, size_t num_boards,
    size_t* solved_count, size_t* error_count) {

  for(int i=0; i<num_boards; i++) {
    // Does the board match the solution?
    if(memcmp(&boards[i], &solutions[i], sizeof(board_t)) == 0) {
      // Record a solved board
      (*solved_count)++;
    } else {
        
      // Make sure the board doesn't have any constraints that rule out
      // values that are supposed to appear in the solution.
      bool valid = true;
      for(int j=0; j<BOARD_DIM * BOARD_DIM; j++) {
        if((boards[i].cells[j] & solutions[i].cells[j]) == 0) {
          valid = false;
        }
      }

      // If the board contains an incorrect constraint, record an error
      if(!valid){
          (*error_count)++;
          continue;
      }

      if(BACKTRACK){
        int ans = backtracking(&boards[i], 3);
        if(memcmp(&boards[i], &solutions[i], sizeof(board_t)) == 0) {
              // Record a solved board
              (*solved_count)++;
        }
        else{
            (*error_count)++;
        }
        
      }
    }
  }
}

int main(int argc, char** argv) {
  
  if(argc != 2) {
    fprintf(stderr, "Usage: %s <input file name>\n", argv[0]);
    exit(1);
  }

  //open input file
  FILE* input = fopen(argv[1], "r");
  if(input == NULL) {
    fprintf(stderr, "Failed to open input file %s.\n", argv[1]);
    perror(NULL);
    exit(2);
  }

  // total boards, boards solved, and incorrect outputs
  size_t num_boards = 0, num_solved = 0, num_errors = 0, solving_time = 0;

  board_t boards[BATCH_SIZE], solutions[BATCH_SIZE];
  
  // position in batch
  size_t batch_pointer = 0;

  // read input file line-by-line
  char* line = NULL;
  size_t line_capacity = 0;
  while(getline(&line, &line_capacity, input) > 0) {
    
    // read the board
    if(!read_board(&boards[batch_pointer], line)) {
      fprintf(stderr, "Skipping invalid board...\n");
      continue;
    }

    // read solution board
    if(!read_board(&solutions[batch_pointer], line + BOARD_DIM * BOARD_DIM + 1)) {
      fprintf(stderr, "Skipping invalid board...\n");
      continue;
    }

    // next batch
    batch_pointer++;

    // increment total count of boards
    num_boards++;

    // completed reading a batch
    if(batch_pointer == BATCH_SIZE) {
      size_t tic = time_ms();

      solve_boards(boards, BATCH_SIZE);

      size_t toc = time_ms();

      solving_time += toc - tic;

      check_solutions(boards, solutions, BATCH_SIZE, &num_solved, &num_errors);

      // reset batch pointer
      batch_pointer = 0;
    }
  }

  // process incomplete batch 
  if(batch_pointer > 0) {
    size_t tic = time_ms();

    solve_boards(boards, batch_pointer);

    size_t toc = time_ms();

    solving_time += toc - tic;

    check_solutions(boards, solutions, batch_pointer, &num_solved, &num_errors);
  }

 
  double seconds = (double)solving_time / 1000;
  double solving_rate = (double)num_solved / seconds;
  
  printf("Total number of boards in the file: %lu\n", num_boards);
  printf("Number of boards solved: %lu\n", num_solved);
  printf("Number of errors(should be zero): %lu\n", num_errors);
  printf("Total solving time: %lums\n", solving_time);
  printf("Solving Rate: %.2f sudoku/second\n", solving_rate);

  return 0;
}