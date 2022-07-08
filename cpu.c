#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#define id(i, j, N) ((i) * N + (j))
#define ROW(p, N) ((p) / N)
#define COL(p, N) ((p) % N)
#define BOX(p, n) ((p) / (n * n * n) * n + ((p) % (n * n)) / n)


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

int backtracking_dfs(int *board, int n, int p, FILE* fp) {
    int N = n * n;
    int i;
    if (p == N * N) {
        // find a new solution, write to file
        print_board(board, N, fp);
        return 1;
    } else {
        if (board[p] == 0) {
            for (i = 0; i < N; ++ i) {
                if (check_partial_board(board, n, p, i + 1)) {
                    board[p] = i + 1;
                    int ret = backtracking_dfs(board, n, p + 1, fp);
                    if (ret) return ret;
                }
            }
            board[p] = 0;
        } else {
            return backtracking_dfs(board, n, p + 1, fp);
        }
        return 0;
    }
}

int backtracking(int *board, int n, FILE* fp) {
    int ans_num = backtracking_dfs(board, n, 0, fp);
    return ans_num;
}

int* read_board(const char* file_name, int* N) {
    FILE* fp = fopen(file_name, "r");
    int* board = NULL;
    
    fscanf(fp, "%d", N);
    
    int total = *N * *N;
    
    board = calloc(total, sizeof(int));
    int i;
    for (i = 0; i < total; ++ i)
        fscanf(fp, "%d", board + i);
    return board;
}


int main(int argc, char* argv[]) {
    
    if (argc < 3) {
        printf("Usage:\n");
        printf("./main <input> <output>\n");
        exit(-1);
    }

    int N;
    int* board = read_board(argv[1], &N);
    
    int n = sqrt((double)N);
    printf("Start to solve a %d x %d Sudoku.\n", N, N);

    FILE* fp = fopen(argv[2], "w");
    int ans;
    ans = backtracking(board, n, fp);
      
    if (ans) printf("An answer is found and saved to %s.\n", argv[2]);
    else printf("No answer is found.\n");

    free(board);
    return 0;
}