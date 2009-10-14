#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
using namespace std;

#define MAX_ITER        25

#define NUM_PATCHES	6
#define MSG_SIZE	1000
#define COMPUTES_PER_P	9

//determine processor locations of computes to which any given processor will send data
void determineDest(int myRank, int numDest, int numPes, int numPatches, int numComputes, int pComputes, int *destinations){
  int startPe = numPes - (((numPatches/numPes*pComputes)*myRank + (numPatches/numPes+1)*min(myRank, numPatches % numPes))%numPes);
  int pe = startPe;
  for (int i = 0; i < numDest; i++){
    destinations[i] = pe;
    pe--;
    if (pe == -1) pe = numPes-1;
  }
}

void sendandreceive(int myRank, int numDest, int numPes, int numPatches, int numComputes, int pComputes, int *destinations, int msgSize, int myComputes){
    /*MPI_Irecv(right_edge_in, blockDimX, MPI_DOUBLE, calc_pe(myRow, wrap_y(myCol+1)), RIGHT, MPI_COMM_WORLD, &req[RIGHT-1]);
    MPI_Irecv(left_edge_in, blockDimX, MPI_DOUBLE, calc_pe(myRow, wrap_y(myCol-1)), LEFT, MPI_COMM_WORLD, &req[LEFT-1]);
    MPI_Irecv(&temperature[blockDimX+1][1], blockDimY, MPI_DOUBLE, calc_pe(wrap_x(myRow+1), myCol), BOTTOM, MPI_COMM_WORLD, &req[BOTTOM-1]);
    MPI_Irecv(&temperature[0][1], blockDimY, MPI_DOUBLE, calc_pe(wrap_x(myRow-1), myCol), TOP, MPI_COMM_WORLD, &req[TOP-1]);


    MPI_Send(left_edge_out, blockDimX, MPI_DOUBLE, calc_pe(myRow, wrap_y(myCol-1)), RIGHT, MPI_COMM_WORLD);
    MPI_Send(right_edge_out, blockDimX, MPI_DOUBLE, calc_pe(myRow, wrap_y(myCol+1)), LEFT, MPI_COMM_WORLD);
    MPI_Send(&temperature[1][1], blockDimY, MPI_DOUBLE, calc_pe(wrap_x(myRow-1), myCol), BOTTOM, MPI_COMM_WORLD);
    MPI_Send(&temperature[blockDimX][1], blockDimY, MPI_DOUBLE, calc_pe(wrap_x(myRow+1), myCol), TOP, MPI_COMM_WORLD);

    MPI_Waitall(4, req, status);
*/
}

int main(int argc, char **argv) {
  int myRank, numPes;
  int numPatches, numComputes;
  int myPatches, pComputes, myComputes, numDest;
  int msgSize;
  int iterations, currIter;
  int* destinations;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Request req[4];
  MPI_Status status[4];

  if (argc > 1) numPatches = atoi(argv[1]);
  else numPatches = NUM_PATCHES;

  if (argc > 2) msgSize = atoi(argv[2]);
  else msgSize = MSG_SIZE;

  if (argc > 3) pComputes = atoi(argv[3]);
  else pComputes = COMPUTES_PER_P;
  
  if (argc > 4) iterations = atoi(argv[4]);
  else iterations = MAX_ITER;

  numComputes = pComputes*numPatches;

  if (myRank == 0){
    printf("Simulating near neighbor communication using:\n");
    printf("%d Processors\n%d Patches\n%d Computes\n%d Message Size\n%d Iterations\n",numPes, numPatches, numComputes, msgSize, iterations);
  }

  //distribute patches to processors in cyclic fashion  
  myPatches = numPatches / numPes;
  if (myRank < numPatches % numPes)
    myPatches++;

  //distribute computes to processors in reverse cyclic fashion
  myComputes = numComputes / numPes;
  if (numPes - myRank < numComputes % numPes)
    myComputes++;
  
  if (myComputes == 0 && myPatches == 0){
    printf("processor %d has no patches or computes\n", myRank);
    MPI_Finalize();
    return 0;
  }
  
  numDest = myPatches*pComputes;
  destinations = new int[myPatches*pComputes];
  
  //determine locations of computes  
  determineDest(myRank, numDest, numPes, numPatches, numComputes, pComputes, destinations);

  double startTimer = MPI_Wtime();
  for (currIter = 0; currIter < iterations; currIter++){
    sendandreceive(myRank, numDest, numPes, numPatches, numComputes, pComputes, destinations, msgSize, myComputes);
    if (myRank == 0 && currIter % 20 == 0){
      printf("%d iterations elapsed, last 20 iterations took %llf seconds\n", currIter, MPI_Wtime() - startTimer);
      startTimer = MPI_Wtime();
    }
  }


  MPI_Finalize();
  return 0;
} /* end function main */

