## Sudoku solver on GPU
### Course Project for the course CS6023(GPU Programming) at IIT Madras
cpu.c -> contains the cpu code for solving sudoku using backtracking.

bt.cu -> contains the gpu code for solving sudoku by the basic backtracking algorithm. initially we expand the search tree on cpu, depending on the 
         input parameter, "expand". Then each board is solved independently by each thread on gpu. 

obt.cu -> contains the gpu code for solving sudoku. we build on the previous method and optimise the backtracking code. each block of N^2 threads works
          on a single board which was expanded on the gpu. 

sudoku.py -> constraint propogation method on cpu. code taken from -> http://norvig.com/sudopy.shtml

sudoku.cu -> constraint propogation method(reference - http://norvig.com/sudoku.html) implemented on gpu. we reduce the search space for each grid and then backtrack.

