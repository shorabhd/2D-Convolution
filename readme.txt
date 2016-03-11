Steps to run the code:

1. Open the terminal
2. Compile the mpi code using "mpicc -c filename.c"
3. Get Executable file by running "mpicc -lm -o filename filename.o"
4. Run the file using "mpirun -n 8 ./filename".(You can change the no. of processors)
5. You can also run the code on GPU nodes using script files. Modify the "filename.sh" script file to change no. of processors.
6. Run the above scripting file on Jarvis by "qsub -pe mpich 2 ./filename.sh".
7. 5 Output files will be created. The output will be in output file and the execution time will be in "run.filename.sh.oJob_Id"
			
					  