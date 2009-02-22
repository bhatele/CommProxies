/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/** \file jacobi2d.c
 *  Author: Abhinav S Bhatele
 *  Date Created: February 19th, 2009
 *
 *
 *    ***********  ^
 *    *		*  |
 *    *		*  |
 *    *		*  X
 *    *		*  |
 *    *		*  |
 *    ***********  ~
 *    <--- Y --->
 *
 *    X: blockDimX, arrayDimX --> wrap_x
 *    Y: arrayDimY
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

/* We want to wrap entries around, and because mod operator % 
 * sometimes misbehaves on negative values. -1 maps to the highest value.*/
#define wrap_x(a)	(((a)+numPes)%numPes)
#define wrap_y(a)	(((a)+arrayDimY)%arrayDimY)

#define MAX_ITER        100
#define TOP             1
#define BOTTOM          2

double startTime;
double endTime;

int main(int argc, char **argv) {
  int myRank, numPes;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Status status;

  int blockDimX, arrayDimX, arrayDimY;

  if (argc != 2 && argc != 3) {
    printf("%s [array_size] \n", argv[0]);
    printf("%s [array_size_X] [array_size_Y] \n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  if(argc == 2) {
    arrayDimY = arrayDimX = atoi(argv[1]);
  }
  else {
    arrayDimX = atoi(argv[1]);
    arrayDimY = atoi(argv[2]);
  }

  if (arrayDimX % numPes != 0) {
    printf("array_size_X % numPes != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  blockDimX = arrayDimX / numPes;

  int iterations = 0, i, j;
  double error = 1.0, max_error = 0.0;

  if(myRank == 0) {
    printf("Running Jacobi on %d processors\n", numPes);
    printf("Array Dimensions: %d %d\n", arrayDimX, arrayDimY);
    printf("Block Dimensions: %d\n", blockDimX);
  }

  double **temperature;
  double **new_temperature;

  /* allocate two dimensional arrays */
  temperature = new double*[blockDimX+2];
  new_temperature = new double*[blockDimX+2];
  for (i=0; i<blockDimX+2; i++) {
    temperature[i] = new double[arrayDimY];
    new_temperature[i] = new double[arrayDimY];
  }
  for(i=0; i<blockDimX+2; i++) {
    for(j=0; j<arrayDimY; j++) {
      temperature[i][j] = 0.5;
      new_temperature[i][j] = 0.5;
    }
  }

  // boundary conditions
  if(myRank < numPes/2) {
    for(i=1; i<=blockDimX; i++)
      temperature[i][0] = 1.0;
  }

  if(myRank == numPes-1) {
    for(j=arrayDimY/2; j<arrayDimY; j++)
      temperature[blockDimX][j] = 0.0;
  }

  startTime = MPI_Wtime();
  while(error > 0.001 && iterations < MAX_ITER) {
    iterations++;

    /* Send my top and bottom edge */
    MPI_Send(&temperature[1][0], arrayDimY, MPI_DOUBLE, wrap_x(myRank-1), BOTTOM, MPI_COMM_WORLD);
    MPI_Send(&temperature[blockDimX][0], arrayDimY, MPI_DOUBLE, wrap_x(myRank+1), TOP, MPI_COMM_WORLD);

    /* Receive my bottom and top edge */
    MPI_Recv(&temperature[blockDimX+1][0], arrayDimY, MPI_DOUBLE, wrap_x(myRank+1), BOTTOM, MPI_COMM_WORLD, &status);
    MPI_Recv(&temperature[0][0], arrayDimY, MPI_DOUBLE, wrap_x(myRank-1), TOP, MPI_COMM_WORLD, &status);

    for(i=1; i<blockDimX+1; i++) {
      for(j=0; j<arrayDimY; j++) {
        /* update my value based on the surrounding values */
        new_temperature[i][j] = (temperature[i-1][j]+temperature[i+1][j]+temperature[i][wrap_y(j-1)]+temperature[i][wrap_y(j+1)]+temperature[i][j]) * 0.2;
      }
    }

    max_error = error = 0.0;
    for(i=1; i<blockDimX+1; i++) {
      for(j=0; j<arrayDimY; j++) {
	error = fabs(new_temperature[i][j] - temperature[i][j]);
	if(error > max_error)
	  max_error = error;
      }
    }
 
    double **tmp;
    tmp = temperature;
    temperature = new_temperature;
    new_temperature = tmp;

    // boundary conditions
    if(myRank < numPes/2) {
      for(i=1; i<=blockDimX; i++)
	temperature[i][0] = 1.0;
    }

    if(myRank == numPes-1) {
      for(j=arrayDimY/2; j<arrayDimY; j++)
	temperature[blockDimX][j] = 0.0;
    }

    //if(myRank == 0) printf("Iteration %d %f %f %f\n", iterations, max_error, temperature[1][0], temperature[1][1]);

    MPI_Allreduce(&max_error, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  } /* end of while loop */

  if(myRank == 0) {
    endTime = MPI_Wtime();
    printf("Completed %d iterations\n", iterations);
    printf("Time elapsed: %f\n", endTime - startTime);
  }

  MPI_Finalize();
  return 0;
} /* end function main */

