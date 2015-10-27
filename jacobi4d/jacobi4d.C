/** \file jacobi4d.C
 *  Author: Nikhil Jain
 *  Date Created: March 12th, 2011
 *
 *        ***********  ^
 *      *         * *  |
 *    ***********   *  |
 *    *		*   *  Y
 *    *		*   *  |
 *    *		*   *  |
 *    *		*   *  ~
 *    *		* *
 *    ***********   Z
 *    <--- X --->
 *<---------------------------t------------------------------>
 *	
 *    X: left, right --> wrap_x
 *    Y: top, bottom --> wrap_y
 *    Z: front, back --> wrap_z
 *    T: forward, backward --> wrap_t	
 *  Four dimensional decomposition of a 4D stencil
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>

/* We want to wrap entries around, and because mod operator % sometimes
 * misbehaves on negative values. -1 maps to the highest value.
 */
#define wrap_x(a)	(((a)+num_blocks_x)%num_blocks_x)
#define wrap_y(a)	(((a)+num_blocks_y)%num_blocks_y)
#define wrap_z(a)	(((a)+num_blocks_z)%num_blocks_z)
#define wrap_t(a)	(((a)+num_blocks_t)%num_blocks_t)

#define index(a,b,c,d)		((a)+(b)*(blockDimX+2)+(c)*(blockDimX+2)*(blockDimY+2)\
				+(d)*(blockDimX+2)*(blockDimY+2)*(blockDimZ+2))
#define calc_pe(a,b,c,d)	((a)+(b)*num_blocks_x+(c)*num_blocks_x*num_blocks_y \
				+(d)*num_blocks_x*num_blocks_y*num_blocks_z)

#define MAX_ITER	10
#define LEFT		1
#define RIGHT		2
#define TOP		3
#define BOTTOM		4
#define FRONT		5
#define BACK		6
#define FORWARD		7
#define BACKWARD	8
#define DIVIDEBY9	0.11111111111111111

double startTime;
double endTime;

int main(int argc, char **argv) {
  int myRank, numPes;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Request sreq[8], rreq[8];

  int blockDimX, blockDimY, blockDimZ, blockDimT;
  int arrayDimX, arrayDimY, arrayDimZ, arrayDimT;
  int noBarrier = 0;

  if (argc != 4 && argc != 10) {
    printf("%s [array_size] [block_size] +[no]barrier\n", argv[0]);
    printf("%s [array_size_X] [array_size_Y] [array_size_Z] [array_size_T] [block_size_X] [block_size_Y] [block_size_Z] [block_size_T] +[no]barrier\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  if(argc == 4) {
    arrayDimT = arrayDimZ = arrayDimY = arrayDimX = atoi(argv[1]);
    blockDimT = blockDimZ = blockDimY = blockDimX = atoi(argv[2]);
    if(strcasecmp(argv[3], "+nobarrier") == 0)
      noBarrier = 1;
    else
      noBarrier = 0;
    if(noBarrier && myRank==0) printf("\nSTENCIL COMPUTATION WITH NO BARRIERS\n");
  }
  else {
    arrayDimX = atoi(argv[1]);
    arrayDimY = atoi(argv[2]);
    arrayDimZ = atoi(argv[3]);
    arrayDimT = atoi(argv[4]);
    blockDimX = atoi(argv[5]);
    blockDimY = atoi(argv[6]);
    blockDimZ = atoi(argv[7]);
    blockDimT = atoi(argv[8]);
    if(strcasecmp(argv[9], "+nobarrier") == 0)
      noBarrier = 1;
    else
      noBarrier = 0;
    if(noBarrier && myRank==0) printf("\nSTENCIL COMPUTATION WITH NO BARRIERS\n");
  }

  if (arrayDimX < blockDimX || arrayDimX % blockDimX != 0) {
    printf("array_size_X % block_size_X != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (arrayDimY < blockDimY || arrayDimY % blockDimY != 0) {
    printf("array_size_Y % block_size_Y != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (arrayDimZ < blockDimZ || arrayDimZ % blockDimZ != 0) {
    printf("array_size_Z % block_size_Z != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (arrayDimT < blockDimT || arrayDimT % blockDimT != 0) {
    printf("array_size_T % block_size_T != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int num_blocks_x = arrayDimX / blockDimX;
  int num_blocks_y = arrayDimY / blockDimY;
  int num_blocks_z = arrayDimZ / blockDimZ;
  int num_blocks_t = arrayDimT / blockDimT;

  int myXcoord = myRank % num_blocks_x;
  int myYcoord = (myRank % (num_blocks_x * num_blocks_y)) / num_blocks_x;
  int myZcoord = (myRank % (num_blocks_x * num_blocks_y * num_blocks_z)) / (num_blocks_x * num_blocks_y);
  int myTcoord = myRank / (num_blocks_x * num_blocks_y * num_blocks_z);

  int iterations = 0, i, j, k, l;
  double error = 1.0, max_error = 0.0;

  if(myRank == 0) {
    printf("Running Jacobi on %d processors with (%d, %d, %d, %d) elements\n", numPes, num_blocks_x, num_blocks_y, num_blocks_z, num_blocks_t);
    printf("Array Dimensions: %d %d %d %d\n", arrayDimX, arrayDimY, arrayDimZ, arrayDimT);
    printf("Block Dimensions: %d %d %d %d\n", blockDimX, blockDimY, blockDimZ, blockDimT);
  }

  double *temperature;
  double *new_temperature;

  /* allocate one dimensional arrays */
  temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2) * (blockDimT+2)];
  new_temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2) * (blockDimT+2)];

  for(l=0; l<blockDimT+2; l++)
    for(k=0; k<blockDimZ+2; k++)
     for(j=0; j<blockDimY+2; j++)
        for(i=0; i<blockDimX+2; i++) {
	  temperature[index(i, j, k, l)] = 0.0;
	}

  /* boundary conditions */
  if(myTcoord == 0 && myZcoord < num_blocks_z/2 && myYcoord < num_blocks_y/2 && myXcoord < num_blocks_x/2) {
    for(k=1; k<=blockDimZ; k++)
      for(j=1; j<=blockDimY; j++)
        for(i=1; i<=blockDimX; i++)
	  temperature[index(i, j, k, 1)] = 1.0;
  }

  if(myZcoord == num_blocks_z-1 && myZcoord >= num_blocks_z/2 && myYcoord >= num_blocks_y/2 && myXcoord >= num_blocks_x/2) {
    for(k=1; k<=blockDimZ; k++)
      for(j=1; j<=blockDimY; j++)
        for(i=1; i<=blockDimX; i++)
	  temperature[index(i, j, k, blockDimT)] = 0.0;
  }

  /* Copy left, right, bottom, top, back, forward and backward  blocks into temporary arrays.*/
  double *left_block_out   = new double[blockDimY*blockDimZ*blockDimT];
  double *right_block_out  = new double[blockDimY*blockDimZ*blockDimT];
  double *left_block_in    = new double[blockDimY*blockDimZ*blockDimT];
  double *right_block_in   = new double[blockDimY*blockDimZ*blockDimT];
  
  double *bottom_block_out = new double[blockDimX*blockDimZ*blockDimT];
  double *top_block_out	   = new double[blockDimX*blockDimZ*blockDimT];
  double *bottom_block_in  = new double[blockDimX*blockDimZ*blockDimT];
  double *top_block_in     = new double[blockDimX*blockDimZ*blockDimT];

  double *front_block_out = new double[blockDimX*blockDimY*blockDimT];
  double *back_block_out  = new double[blockDimX*blockDimY*blockDimT];
  double *front_block_in  = new double[blockDimX*blockDimY*blockDimT];
  double *back_block_in   = new double[blockDimX*blockDimY*blockDimT];

  double *forward_block_out   = new double[blockDimX*blockDimY*blockDimZ];
  double *backward_block_out  = new double[blockDimX*blockDimY*blockDimZ];
  double *forward_block_in    = new double[blockDimX*blockDimY*blockDimZ];
  double *backward_block_in   = new double[blockDimX*blockDimY*blockDimZ];

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Pcontrol(1);
  startTime = MPI_Wtime();

  while(/*error > 0.001 &&*/ iterations < MAX_ITER) {
    iterations++;

    /* Copy different planes into buffers */
    for(l=0; l<blockDimT; ++l)
      for(k=0; k<blockDimZ; ++k)
	for(j=0; j<blockDimY; ++j) {
	  left_block_out[l*(blockDimY*blockDimZ)+k*blockDimY+j] = temperature[index(1, j+1, k+1, l+1)];
	  right_block_out[l*(blockDimY*blockDimZ)+k*blockDimY+j] = temperature[index(blockDimX, j+1, k+1, l+1)];
      }

    for(l=0; l<blockDimT; ++l)
      for(k=0; k<blockDimZ; ++k)
	for(i=0; i<blockDimX; ++i) {
	  top_block_out[l*(blockDimX*blockDimZ)+k*blockDimX+i] = temperature[index(i+1, 1, k+1, l+1)];
	  bottom_block_out[l*(blockDimX*blockDimZ)+k*blockDimX+i] = temperature[index(i+1, blockDimY, k+1, l+1)];
      }

    for(l=0; l<blockDimT; ++l)
      for(j=0; j<blockDimY; ++j)
	for(i=0; i<blockDimX; ++i) {
	  front_block_out[l*(blockDimX*blockDimY)+j*blockDimX+i] = temperature[index(i+1, j+1, 1, l+1)];
	  back_block_out[l*(blockDimX*blockDimY)+j*blockDimX+i] = temperature[index(i+1, j+1, blockDimZ, l+1)];
      }

    for(k=0; k<blockDimZ; ++k)
      for(j=0; j<blockDimY; ++j)
	for(i=0; i<blockDimX; ++i) {
	  forward_block_out[k*(blockDimX*blockDimY)+j*blockDimX+i] = temperature[index(i+1, j+1, k+1, 1)];
	  backward_block_out[k*(blockDimX*blockDimY)+j*blockDimX+i] = temperature[index(i+1, j+1, k+1, blockDimT)];
      }

    /* Receive my right, left, top, bottom, back, front, forward and backward blocks */
    MPI_Irecv(right_block_in, blockDimY*blockDimZ*blockDimT, MPI_DOUBLE, calc_pe(wrap_x(myXcoord+1), myYcoord, myZcoord, myTcoord), RIGHT, MPI_COMM_WORLD, &rreq[RIGHT-1]);
    MPI_Irecv(left_block_in, blockDimY*blockDimZ*blockDimT, MPI_DOUBLE, calc_pe(wrap_x(myXcoord-1), myYcoord, myZcoord, myTcoord), LEFT, MPI_COMM_WORLD, &rreq[LEFT-1]);
    MPI_Irecv(top_block_in, blockDimX*blockDimZ*blockDimT, MPI_DOUBLE, calc_pe(myXcoord, wrap_y(myYcoord+1), myZcoord, myTcoord), TOP, MPI_COMM_WORLD, &rreq[TOP-1]);
    MPI_Irecv(bottom_block_in, blockDimX*blockDimZ*blockDimT, MPI_DOUBLE, calc_pe(myXcoord, wrap_y(myYcoord-1), myZcoord, myTcoord), BOTTOM, MPI_COMM_WORLD, &rreq[BOTTOM-1]);
    MPI_Irecv(front_block_in, blockDimX*blockDimY*blockDimT, MPI_DOUBLE, calc_pe(myXcoord, myYcoord, wrap_z(myZcoord+1), myTcoord), FRONT, MPI_COMM_WORLD, &rreq[FRONT-1]);
    MPI_Irecv(back_block_in, blockDimX*blockDimY*blockDimT, MPI_DOUBLE, calc_pe(myXcoord, myYcoord, wrap_z(myZcoord-1), myTcoord), BACK, MPI_COMM_WORLD, &rreq[BACK-1]);
    MPI_Irecv(forward_block_in, blockDimX*blockDimY*blockDimZ, MPI_DOUBLE, calc_pe(myXcoord, myYcoord, myZcoord, wrap_t(myTcoord+1)), FORWARD, MPI_COMM_WORLD, &rreq[FORWARD-1]);
    MPI_Irecv(backward_block_in, blockDimX*blockDimY*blockDimZ, MPI_DOUBLE, calc_pe(myXcoord, myYcoord, myZcoord, wrap_t(myTcoord-1)), BACKWARD, MPI_COMM_WORLD, &rreq[BACKWARD-1]);

    /* Send my left, right, bottom, top, front, back, forward  and backward blocks */
    MPI_Isend(left_block_out, blockDimY*blockDimZ*blockDimT, MPI_DOUBLE, calc_pe(wrap_x(myXcoord-1), myYcoord, myZcoord, myTcoord), RIGHT, MPI_COMM_WORLD, &sreq[RIGHT-1]);
    MPI_Isend(right_block_out, blockDimY*blockDimZ*blockDimT, MPI_DOUBLE, calc_pe(wrap_x(myXcoord+1), myYcoord, myZcoord, myTcoord), LEFT, MPI_COMM_WORLD, &sreq[LEFT-1]);
    MPI_Isend(bottom_block_out, blockDimX*blockDimZ*blockDimT, MPI_DOUBLE, calc_pe(myXcoord, wrap_y(myYcoord-1), myZcoord, myTcoord), TOP, MPI_COMM_WORLD, &sreq[TOP-1]);
    MPI_Isend(top_block_out, blockDimX*blockDimZ*blockDimT, MPI_DOUBLE, calc_pe(myXcoord, wrap_y(myYcoord+1), myZcoord, myTcoord), BOTTOM, MPI_COMM_WORLD, &sreq[BOTTOM-1]);
    MPI_Isend(back_block_out, blockDimX*blockDimY*blockDimT, MPI_DOUBLE, calc_pe(myXcoord, myYcoord, wrap_z(myZcoord-1), myTcoord), FRONT, MPI_COMM_WORLD, &sreq[FRONT-1]);
    MPI_Isend(front_block_out, blockDimX*blockDimY*blockDimT, MPI_DOUBLE, calc_pe(myXcoord, myYcoord, wrap_z(myZcoord+1), myTcoord), BACK, MPI_COMM_WORLD, &sreq[BACK-1]);
    MPI_Isend(backward_block_out, blockDimX*blockDimY*blockDimZ, MPI_DOUBLE, calc_pe(myXcoord, myYcoord, myZcoord, wrap_t(myTcoord-1)), FORWARD, MPI_COMM_WORLD, &sreq[FORWARD-1]);
    MPI_Isend(forward_block_out, blockDimX*blockDimY*blockDimZ, MPI_DOUBLE, calc_pe(myXcoord, myYcoord, myZcoord, wrap_t(myTcoord+1)), BACKWARD, MPI_COMM_WORLD, &sreq[BACKWARD-1]);

    MPI_Waitall(8, rreq, MPI_STATUSES_IGNORE);
    MPI_Waitall(8, sreq, MPI_STATUSES_IGNORE);

    /* Copy buffers into ghost layers */
    for(l=0; l<blockDimT; ++l)
      for(k=0; k<blockDimZ; ++k)
        for(j=0; j<blockDimY; ++j) {
	  temperature[index(0, j+1, k+1, l+1)] = left_block_in[l*blockDimZ*blockDimY+k*blockDimY+j];
      }
    for(l=0; l<blockDimT; ++l)
      for(k=0; k<blockDimZ; ++k)
	for(j=0; j<blockDimY; ++j) 
	  temperature[index(blockDimX+1, j+1, k+1, l+1)] = right_block_in[l*blockDimZ*blockDimY+k*blockDimY+j];
      }
    for(l=0; l<blockDimT; ++l)
      for(k=0; k<blockDimZ; ++k)
	for(i=0; i<blockDimX; ++i) {
	  temperature[index(i+1, 0, k+1, l+1)] = bottom_block_in[l*blockDimZ*blockDimX+k*blockDimX+i];
      }
    for(l=0; l<blockDimT; ++l)
      for(k=0; k<blockDimZ; ++k)
	for(i=0; i<blockDimX; ++i) {
	  temperature[index(i+1, blockDimY+1, k+1, l+1)] = top_block_in[l*blockDimZ*blockDimX+k*blockDimX+i];
      }
    for(l=0; l<blockDimT; ++l)
      for(j=0; j<blockDimY; ++j) 
	for(i=0; i<blockDimX; ++i) {
	  temperature[index(i+1, j+1, 0, l+1)] = back_block_in[l*blockDimY*blockDimX+j*blockDimX+i];
      }
    for(l=0; l<blockDimT; ++l)
      for(j=0; j<blockDimY; ++j) 
	for(i=0; i<blockDimX; ++i) {
	  temperature[index(i+1, j+1, blockDimZ+1, l+1)] = front_block_in[l*blockDimY*blockDimX+j*blockDimX+i];
      }
    for(k=0; k<blockDimT; ++k)
      for(j=0; j<blockDimY; ++j) 
	for(i=0; i<blockDimX; ++i) {
	  temperature[index(i+1, j+1, k+1, 0)] = back_block_in[k*blockDimY*blockDimX+j*blockDimX+i];
      }
    for(k=0; k<blockDimT; ++k)
      for(j=0; j<blockDimY; ++j) 
	for(i=0; i<blockDimX; ++i) {
	  temperature[index(i+1, j+1, k+1, blockDimT+1)] = front_block_in[k*blockDimY*blockDimX+j*blockDimX+i];


    /* update my value based on the surrounding values */
    for(l=1; l<blockDimT+1; ++l)
      for(k=1; k<blockDimZ+1; k++)
	for(j=1; j<blockDimY+1; j++)
	  for(i=1; i<blockDimX+1; i++) {
	    new_temperature[index(i, j, k, l)] = (temperature[index(i-1, j, k, l)]
	                                    +  temperature[index(i+1, j, k, l)]
		                            +  temperature[index(i, j-1, k, l)]
		                            +  temperature[index(i, j+1, k, l)]
		                            +  temperature[index(i, j, k-1, l)]
		                            +  temperature[index(i, j, k+1, l)]
		                            +  temperature[index(i, j, k, l-1)]
		                            +  temperature[index(i, j, k, l+1)]
			                    +  temperature[index(i, j, k, l)] ) * DIVIDEBY9;
	}

    max_error = error = 0.0;
    for(l=1; l<blockDimZ+1; l++)
      for(k=1; k<blockDimZ+1; k++)
	for(j=1; j<blockDimY+1; j++)
	  for(i=1; i<blockDimX+1; i++) {
	    error = fabs(new_temperature[index(i, j, k, l)] - temperature[index(i, j, k, l)]);
	  if(error > max_error)
	    max_error = error;
	}
 
    double *tmp;
    tmp = temperature;
    temperature = new_temperature;
    new_temperature = tmp;

    /* boundary conditions */
    if(myTcoord == 0 && myZcoord < num_blocks_z/2 && myYcoord < num_blocks_y/2 && myXcoord < num_blocks_x/2) {
      for(k=1; k<=blockDimZ; k++)
        for(j=1; j<=blockDimY; j++)
	  for(i=1; i<=blockDimX; i++)
	    temperature[index(i, j, k, 1)] = 1.0;
    }

    if(myZcoord == num_blocks_z-1 && myZcoord >= num_blocks_z/2 && myYcoord >= num_blocks_y/2 && myXcoord >= num_blocks_x/2) {
      for(k=1; k<=blockDimZ; k++)
	for(j=1; j<=blockDimY; j++)
	  for(i=1; i<=blockDimX; i++)
	    temperature[index(i, j, k, blockDimT)] = 0.0;
    }

    // if(myRank == 0) printf("Iteration %d %f\n", iterations, max_error);
    if(noBarrier == 0) MPI_Allreduce(&max_error, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  } /* end of while loop */

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Pcontrol(0);

  if(myRank == 0) {
    endTime = MPI_Wtime();
    printf("Completed %d iterations\n", iterations);
    printf("Time elapsed per iteration: %f\n", (endTime - startTime)/(MAX_ITER));
  }

  MPI_Finalize();
  return 0;
} /* end function main */

