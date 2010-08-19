/** \file fft1d.c
 *  Author: Abhinav S Bhatele
 *  Date Created: August 19th, 2010
 *  E-mail: bhatele@illinois.edu
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

double startTime;
double endTime;

int main(int argc, char **argv) {
  int myRank, numPes;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Request req;
  MPI_Status status;

  int numElements, numSteps, iteration = 0;

  if (argc != 2) {
    printf("%s [number of elements]\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  } else {
    numElements = atoi(argv[1]);
  }

  numSteps = (int) log2((float)numPes);

  if(myRank == 0) {
    printf("Running 1D FFT on %d processors ...\n", numPes);
  }

  int sendto;
  double val = myRank;

  while(iteration < numSteps) {
    if( (myRank/(int)pow(2, iteration)) % 2 == 0)
      sendto = myRank + pow(2, iteration);
    else
      sendto = myRank - pow(2, iteration);

    /* Receive my element */
    MPI_Irecv(&val, 1, MPI_DOUBLE, sendto, myRank, MPI_COMM_WORLD, &req);

    /* Send my element*/
    MPI_Send(&val, 1, MPI_DOUBLE, sendto, myRank, MPI_COMM_WORLD);

    printf("[%d] Rank %d sending to rank %d\n", iteration, myRank, sendto);
    iteration++;
    MPI_Barrier(MPI_COMM_WORLD);
  } /* end of while loop */

  if(myRank == 0) {
    endTime = MPI_Wtime();
    printf("1D FFT completed\n");
    printf("Time elapsed: %f\n", (endTime - startTime));
  }

  MPI_Finalize();
  return 0;
} /* end function main */

