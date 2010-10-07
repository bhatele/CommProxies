/** \file allreduce.c
 *  Author: Abhinav S Bhatele
 *  Date Created: July 31st, 2009
 *  E-mail: bhatele@illinois.edu
 *
 *  This program does an allreduce.
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

// Minimum message size (bytes)
#define MIN_MSG_SIZE 4

// Maximum message size (bytes)
#define MAX_MSG_SIZE (1024 * 1024)

#define NUM_MSGS 100

int main(int argc, char *argv[]) {
  int numprocs, myrank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int val = myrank;
  int sum = 0, total = 0, i;
  double sendTime, recvTime;

  // warm-up
  for (i=0; i<10; i++) {
    MPI_Allreduce(&val, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    total += sum;
  }

  sendTime = MPI_Wtime();
  // if(myrank == 0) BgPrintf("Start of loop at %f\n");
  for (i=0; i<NUM_MSGS; i++) {
    MPI_Allreduce(&val, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    total += sum;
  }
  // if(myrank == 0) BgPrintf("End of loop at %f\n");
  recvTime = (MPI_Wtime() - sendTime) / NUM_MSGS;

  // cool down
  for (i=0; i<10; i++) {
    MPI_Allreduce(&val, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    total += sum;
  }

  if(myrank == 0) printf("%d %g\n", numprocs, recvTime);

  MPI_Finalize();
  return 0;
}

