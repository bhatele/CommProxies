/** \file 14ptStencil.C
 *  Author: Abhinav S Bhatele
 *  Date Created: December 19th, 2010
 *
 *  Author: Michael Robson
 *  Date Modified: April 1st, 2013
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
 *
 *    X: left, right --> wrap_x
 *    Y: top, bottom --> wrap_y
 *    Z: front, back --> wrap_z
 *
 *  Three dimensional decomposition of a 3D stencil
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

#define index(a,b,c)	((a)+(b)*(blockDimX+2)+(c)*(blockDimX+2)*(blockDimY+2))
#define calc_pe(a,b,c)	((a)+(b)*num_blocks_x+(c)*num_blocks_x*num_blocks_y)

#define MAX_ITER		10
#define LEFT			1
#define RIGHT			2
#define TOP			3
#define BOTTOM			4
#define FRONT			5
#define BACK			6
#define DIVIDEBY7		0.14285714285714285714

#define TOP_FRONT_RIGHT		7
#define TOP_FRONT_LEFT		8
#define TOP_BACK_RIGHT		9
#define TOP_BACK_LEFT		10
#define BOTTOM_FRONT_RIGHT	11
#define BOTTOM_FRONT_LEFT	12
#define BOTTOM_BACK_RIGHT	13
#define BOTTOM_BACK_LEFT	14

double startTime;
double endTime;

int main(int argc, char **argv) {
  int myRank, numPes;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Request rreq[14], sreq[14];

  int blockDimX, blockDimY, blockDimZ;
  int arrayDimX, arrayDimY, arrayDimZ;
  int noBarrier = 0;

  int messageSize;

  if (argc != 5 && argc != 9) {
    printf("%s [array_size] [block_size] [message_size] +[no]barrier\n", argv[0]);
    printf("%s [array_size_X] [array_size_Y] [array_size_Z] [block_size_X] [block_size_Y] [block_size_Z] [message_size] +[no]barrier\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  if(argc == 5) {
    arrayDimZ = arrayDimY = arrayDimX = atoi(argv[1]);
    blockDimZ = blockDimY = blockDimX = atoi(argv[2]);
    messageSize = atoi(argv[3]);
    if(strcasecmp(argv[4], "+nobarrier") == 0)
      noBarrier = 1;
    else
      noBarrier = 0;
    if(noBarrier && myRank==0) printf("\nSTENCIL COMPUTATION WITH NO BARRIERS\n");
  }
  else {
    arrayDimX = atoi(argv[1]);
    arrayDimY = atoi(argv[2]);
    arrayDimZ = atoi(argv[3]);
    blockDimX = atoi(argv[4]);
    blockDimY = atoi(argv[5]);
    blockDimZ = atoi(argv[6]);
    messageSize = atoi(argv[7]);
    if(strcasecmp(argv[8], "+nobarrier") == 0)
      noBarrier = 1;
    else
      noBarrier = 0;
    if(noBarrier && myRank==0) printf("\nSTENCIL COMPUTATION WITH NO BARRIERS\n");
  }

  if (arrayDimX < blockDimX || arrayDimX % blockDimX != 0) {
    printf("array_size_X %% block_size_X != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (arrayDimY < blockDimY || arrayDimY % blockDimY != 0) {
    printf("array_size_Y %% block_size_Y != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (arrayDimZ < blockDimZ || arrayDimZ % blockDimZ != 0) {
    printf("array_size_Z %% block_size_Z != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int num_blocks_x = arrayDimX / blockDimX;
  int num_blocks_y = arrayDimY / blockDimY;
  int num_blocks_z = arrayDimZ / blockDimZ;

  int myXcoord = myRank % num_blocks_x;
  int myYcoord = (myRank % (num_blocks_x * num_blocks_y)) / num_blocks_x;
  int myZcoord = myRank / (num_blocks_x * num_blocks_y);

  int iterations = 0, i, j, k;
  double error = 1.0, max_error = 0.0;

  if(myRank == 0) {
    printf("Running Jacobi on %d processors with (%d, %d, %d) elements and %d B messages\n", numPes, num_blocks_x, num_blocks_y, num_blocks_z, messageSize);
    printf("Array Dimensions: %d %d %d\n", arrayDimX, arrayDimY, arrayDimZ);
    printf("Block Dimensions: %d %d %d\n", blockDimX, blockDimY, blockDimZ);
  }

  /* Copy left, right, bottom, top, front and back  planes into temporary arrays. */

  double *left_plane_out   = new double[messageSize];
  double *right_plane_out  = new double[messageSize];
  double *left_plane_in    = new double[messageSize];
  double *right_plane_in   = new double[messageSize];

  double *bottom_plane_out = new double[messageSize];
  double *top_plane_out	   = new double[messageSize];
  double *bottom_plane_in  = new double[messageSize];
  double *top_plane_in     = new double[messageSize];

  double *back_plane_out    = new double[messageSize];
  double *front_plane_out   = new double[messageSize];
  double *back_plane_in     = new double[messageSize];
  double *front_plane_in    = new double[messageSize];

  /* Copy bottom back left, back right, front left, and front right planes (from corners) */ 

  double *bottom_back_left_plane_out   = new double[messageSize];
  double *bottom_back_right_plane_out  = new double[messageSize];
  double *bottom_back_left_plane_in    = new double[messageSize];
  double *bottom_back_right_plane_in   = new double[messageSize];

  double *bottom_front_left_plane_out   = new double[messageSize];
  double *bottom_front_right_plane_out  = new double[messageSize];
  double *bottom_front_left_plane_in    = new double[messageSize];
  double *bottom_front_right_plane_in   = new double[messageSize];

  /* Copy my top back left, back right, front left, and front right planes (from corners) */ 

  double *top_back_left_plane_out   = new double[messageSize];
  double *top_back_right_plane_out  = new double[messageSize];
  double *top_back_left_plane_in    = new double[messageSize];
  double *top_back_right_plane_in   = new double[messageSize];

  double *top_front_left_plane_out   = new double[messageSize];
  double *top_front_right_plane_out  = new double[messageSize];
  double *top_front_left_plane_in    = new double[messageSize];
  double *top_front_right_plane_in   = new double[messageSize];

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Pcontrol(1);
  startTime = MPI_Wtime();

  while(/*error > 0.001 &&*/ iterations < MAX_ITER) {
    iterations++;

    /* Receive my right, left, top, bottom, back and front planes */
    MPI_Irecv(right_plane_in, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord+1), myYcoord, myZcoord), RIGHT, MPI_COMM_WORLD, &rreq[RIGHT-1]);
    MPI_Irecv(left_plane_in, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord-1), myYcoord, myZcoord), LEFT, MPI_COMM_WORLD, &rreq[LEFT-1]);
    MPI_Irecv(top_plane_in, messageSize, MPI_DOUBLE, calc_pe(myXcoord, wrap_y(myYcoord+1), myZcoord), TOP, MPI_COMM_WORLD, &rreq[TOP-1]);
    MPI_Irecv(bottom_plane_in, messageSize, MPI_DOUBLE, calc_pe(myXcoord, wrap_y(myYcoord-1), myZcoord), BOTTOM, MPI_COMM_WORLD, &rreq[BOTTOM-1]);
    MPI_Irecv(front_plane_in, messageSize, MPI_DOUBLE, calc_pe(myXcoord, myYcoord, wrap_z(myZcoord+1)), FRONT, MPI_COMM_WORLD, &rreq[FRONT-1]);
    MPI_Irecv(back_plane_in, messageSize, MPI_DOUBLE, calc_pe(myXcoord, myYcoord, wrap_z(myZcoord-1)), BACK, MPI_COMM_WORLD, &rreq[BACK-1]);

    /* Recieve my top front right, front left, back right, and  back left planes (from corners) */ 
    MPI_Irecv(top_front_right_plane_in, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord+1), wrap_y(myYcoord+1), wrap_z(myZcoord+1)), TOP_FRONT_RIGHT, MPI_COMM_WORLD, &rreq[TOP_FRONT_RIGHT-1]);
    MPI_Irecv(top_front_left_plane_in, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord-1), wrap_y(myYcoord+1), wrap_z(myZcoord+1)), TOP_FRONT_LEFT, MPI_COMM_WORLD, &rreq[TOP_FRONT_LEFT-1]);
    MPI_Irecv(top_back_right_plane_in, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord+1), wrap_y(myYcoord+1), wrap_z(myZcoord-1)), TOP_BACK_RIGHT, MPI_COMM_WORLD, &rreq[TOP_BACK_RIGHT-1]);
    MPI_Irecv(top_back_left_plane_in, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord-1), wrap_y(myYcoord+1), wrap_z(myZcoord-1)), TOP_BACK_LEFT, MPI_COMM_WORLD, &rreq[TOP_BACK_LEFT-1]);

    /* Recieve my bottom back right, back left, front right, and front left planes (from corners) */ 
    MPI_Irecv(bottom_front_right_plane_in, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord+1), wrap_y(myYcoord-1), wrap_z(myZcoord+1)), BOTTOM_FRONT_RIGHT, MPI_COMM_WORLD, &rreq[BOTTOM_FRONT_RIGHT-1]);
    MPI_Irecv(bottom_front_left_plane_in, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord-1), wrap_y(myYcoord-1), wrap_z(myZcoord+1)), BOTTOM_FRONT_LEFT, MPI_COMM_WORLD, &rreq[BOTTOM_FRONT_LEFT-1]);
    MPI_Irecv(bottom_back_right_plane_in, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord+1), wrap_y(myYcoord-1), wrap_z(myZcoord-1)), BOTTOM_BACK_RIGHT, MPI_COMM_WORLD, &rreq[BOTTOM_BACK_RIGHT-1]);
    MPI_Irecv(bottom_back_left_plane_in, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord-1), wrap_y(myYcoord-1), wrap_z(myZcoord-1)), BOTTOM_BACK_LEFT, MPI_COMM_WORLD, &rreq[BOTTOM_BACK_LEFT-1]);

    /* Send my left, right, bottom, top, front and back planes */
    MPI_Isend(left_plane_out, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord-1), myYcoord, myZcoord), RIGHT, MPI_COMM_WORLD, &sreq[0]);
    MPI_Isend(right_plane_out, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord+1), myYcoord, myZcoord), LEFT, MPI_COMM_WORLD, &sreq[1]);
    MPI_Isend(bottom_plane_out, messageSize, MPI_DOUBLE, calc_pe(myXcoord, wrap_y(myYcoord-1), myZcoord), TOP, MPI_COMM_WORLD, &sreq[2]);
    MPI_Isend(top_plane_out, messageSize, MPI_DOUBLE, calc_pe(myXcoord, wrap_y(myYcoord+1), myZcoord), BOTTOM, MPI_COMM_WORLD, &sreq[3]);
    MPI_Isend(back_plane_out, messageSize, MPI_DOUBLE, calc_pe(myXcoord, myYcoord, wrap_z(myZcoord-1)), FRONT, MPI_COMM_WORLD, &sreq[4]);
    MPI_Isend(front_plane_out, messageSize, MPI_DOUBLE, calc_pe(myXcoord, myYcoord, wrap_z(myZcoord+1)), BACK, MPI_COMM_WORLD, &sreq[5]);

    /* Send my bottom back left, back right, front left, and front right planes (from corners) */ 
    MPI_Isend(bottom_back_left_plane_out, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord-1), wrap_y(myYcoord-1), wrap_z(myZcoord-1)), TOP_FRONT_RIGHT, MPI_COMM_WORLD, &sreq[6]);
    MPI_Isend(bottom_back_right_plane_out, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord+1), wrap_y(myYcoord-1), wrap_z(myZcoord-1)), TOP_FRONT_LEFT, MPI_COMM_WORLD, &sreq[7]);
    MPI_Isend(bottom_front_left_plane_out, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord-1), wrap_y(myYcoord-1), wrap_z(myZcoord+1)), TOP_BACK_RIGHT, MPI_COMM_WORLD, &sreq[8]);
    MPI_Isend(bottom_front_right_plane_out, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord+1), wrap_y(myYcoord-1), wrap_z(myZcoord+1)), TOP_BACK_LEFT, MPI_COMM_WORLD, &sreq[9]);

    /* Send my top back left, back right, front left, and front right planes (from corners) */ 
    MPI_Isend(top_back_left_plane_out, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord-1), wrap_y(myYcoord+1), wrap_z(myZcoord-1)), BOTTOM_FRONT_RIGHT, MPI_COMM_WORLD, &sreq[10]);
    MPI_Isend(top_back_right_plane_out, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord+1), wrap_y(myYcoord+1), wrap_z(myZcoord-1)), BOTTOM_FRONT_LEFT, MPI_COMM_WORLD, &sreq[11]);
    MPI_Isend(top_front_left_plane_out, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord-1), wrap_y(myYcoord+1), wrap_z(myZcoord+1)), BOTTOM_BACK_RIGHT, MPI_COMM_WORLD, &sreq[12]);
    MPI_Isend(top_front_right_plane_out, messageSize, MPI_DOUBLE, calc_pe(wrap_x(myXcoord+1), wrap_y(myYcoord+1), wrap_z(myZcoord+1)), BOTTOM_BACK_LEFT, MPI_COMM_WORLD, &sreq[13]);

    MPI_Waitall(14, rreq, MPI_STATUSES_IGNORE);
    MPI_Waitall(14, sreq, MPI_STATUSES_IGNORE);

  } /* end of while loop */

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Pcontrol(0);

  if(myRank == 0) {
    endTime = MPI_Wtime();
    printf("Completed %d iterations\n", iterations);
    printf("Time elapsed per iteration: %f\n", (endTime - startTime)/(MAX_ITER-5));
  }

  delete [] left_plane_out; delete [] right_plane_out;
  delete [] left_plane_in; delete [] right_plane_in;
  delete [] bottom_plane_out; delete [] top_plane_out;
  delete [] bottom_plane_in; delete [] top_plane_in;
  delete [] back_plane_out; delete [] front_plane_out;
  delete [] back_plane_in; delete [] front_plane_in;
  delete [] bottom_back_left_plane_out; delete [] bottom_back_right_plane_out;
  delete [] bottom_front_left_plane_in; delete [] bottom_front_right_plane_in;
  delete [] top_back_left_plane_out; delete [] top_back_right_plane_out;
  delete [] top_front_left_plane_in; delete [] top_front_right_plane_in;

  MPI_Finalize();
  return 0;
} /* end function main */

