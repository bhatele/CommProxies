#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
using namespace std;

#define MAX_ITER        200

#define NUM_PATCHES	6
#define MSG_SIZE	1000
#define COMPUTES_PER_P	9

//determine processor locations of computes to which any given processor will send data
void determineDest(int myRank, int numDest, int numPes, int numPatches, int numComputes, int pComputes, int *destinations, int k){
  //int startPe = numPes - (((numPatches/numPes*pComputes)*myRank + (numPatches/numPes+1)*min(myRank, numPatches % numPes))%numPes);
  int startPe = (myRank + numPes*50 - (numDest/2)*k)%numPes;
  int pe = startPe;
  for (int i = 0; i < numDest; i++){
    destinations[i] = pe;
    pe+=k;
    if (pe >= numPes) pe -= numPes;
  }
}

double sendandreceive(int myRank, int numDest, int numPes, int *destinations, int msgSize, int myComputes, int **recvbuffer, int **sendbuffer, MPI_Request *req, MPI_Status *stat){
  printf("sending with message size = %d\n", msgSize);
  double sendtime = MPI_Wtime();
  for ( int i = 0; i < numDest; i++){
    MPI_Irecv(recvbuffer[i], msgSize, MPI_INT, destinations[i], myRank*numPes+destinations[i], MPI_COMM_WORLD, &req[i]);
    MPI_Send(sendbuffer[i], msgSize, MPI_INT, destinations[i], destinations[i]*numPes + myRank, MPI_COMM_WORLD);
  }
  sendtime = MPI_Wtime() - sendtime;
  MPI_Waitall(numDest, req, stat);
  return sendtime;
}

double recvfirst(int myRank, int numDest, int numPes, int *destinations, int msgSize, int myComputes, int **recvbuffer, int **sendbuffer, MPI_Request *req, MPI_Status *stat){
  for ( int i = 0; i < numDest; i++){
    MPI_Irecv(recvbuffer[i], msgSize, MPI_INT, destinations[i], myRank*numPes+destinations[i], MPI_COMM_WORLD, &req[i]);
  }
  double sendtime = MPI_Wtime();
  for (int i = 0; i < numDest; i++){
    MPI_Send(sendbuffer[i], msgSize, MPI_INT, destinations[i], destinations[i]*numPes + myRank, MPI_COMM_WORLD);
  }
  sendtime = MPI_Wtime() - sendtime;
  MPI_Waitall(numDest, req, stat);
  return sendtime;
}

double sendfirst(int myRank, int numDest, int numPes, int *destinations, int msgSize, int myComputes, int **recvbuffer, int **sendbuffer, MPI_Request *req, MPI_Status *stat){
  double sendtime = MPI_Wtime();
  for (int i = 0; i < numDest; i++){
    MPI_Send(sendbuffer[i], msgSize, MPI_INT, destinations[i], destinations[i]*numPes + myRank, MPI_COMM_WORLD);
  }
  sendtime = MPI_Wtime() - sendtime;
  for ( int i = 0; i < numDest; i++){
    MPI_Irecv(recvbuffer[i], msgSize, MPI_INT, destinations[i], myRank*numPes+destinations[i], MPI_COMM_WORLD, &req[i]);
  }
  MPI_Waitall(numDest, req, stat);
  return sendtime;
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
  
  if (argc > 1) numPatches = atoi(argv[1]);
  else numPatches = NUM_PATCHES;

  if (argc > 2) pComputes = atoi(argv[2]);
  else pComputes = COMPUTES_PER_P;
  
  if (argc > 3) msgSize = atoi(argv[3]);
  else msgSize = MSG_SIZE;
  
  if (argc > 4) iterations = atoi(argv[4]);
  else iterations = MAX_ITER;

  numPatches = numPes;
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
  MPI_Request *req = new MPI_Request[numDest];
  MPI_Status *stat = new MPI_Status[numDest];
  
  //determine locations of computes  
  determineDest(myRank, numDest, numPes, numPatches, numComputes, pComputes, destinations,3);
  int **recvbuffer = new int*[numDest];
  int **sendbuffer = new int*[numDest];
  for ( int i = 0; i < numDest; i++){
    sendbuffer[i] = new int[msgSize];
    recvbuffer[i] = new int[msgSize];
  }

  double startTimer = MPI_Wtime();
  double sendTime = 0;
  /*if (myRank == 0) printf("Doing loop Irecv Send loop++ iterations\n");
  for (currIter = 0; currIter < iterations; currIter++){
    sendTime += sendandreceive(myRank, numDest, numPes, destinations, msgSize, myComputes, recvbuffer, sendbuffer, req, stat);
    if (myRank == 0 && currIter % 20 == 0){
      printf("%d iterations elapsed, last 20 iterations took %lf seconds, with a total sendTime of %lf\n", currIter, MPI_Wtime() - startTimer, sendTime);
      startTimer = MPI_Wtime();
      sendTime = 0;
    }
  }*/
  
  startTimer = MPI_Wtime();
  sendTime = 0;
  if (myRank == 0) printf("Doing recvfirst iterations\n");
  for (currIter = 0; currIter < iterations; currIter++){
    sendTime += recvfirst(myRank, numDest, numPes, destinations, msgSize, myComputes, recvbuffer, sendbuffer, req, stat);
    if (myRank == 0 && currIter % 20 == 0){
      printf("k=3 %d iterations elapsed, last 20 iterations took %lf seconds, with a total sendTime of %lf\n", currIter, MPI_Wtime() - startTimer, sendTime);
      startTimer = MPI_Wtime();
      sendTime = 0;
    }
  }
  

  determineDest(myRank, numDest, numPes, numPatches, numComputes, pComputes, destinations,5);

  startTimer = MPI_Wtime();
  sendTime = 0;
  if (myRank == 0) printf("Doing recvfirst iterations\n");
  for (currIter = 0; currIter < iterations; currIter++){
    sendTime += recvfirst(myRank, numDest, numPes, destinations, msgSize, myComputes, recvbuffer, sendbuffer, req, stat);
    if (myRank == 0 && currIter % 20 == 0){
      printf("k=5 %d iterations elapsed, last 20 iterations took %lf seconds, with a total sendTime of %lf\n", currIter, MPI_Wtime() - startTimer, sendTime);
      startTimer = MPI_Wtime();
      sendTime = 0;
    }
  }
  

  determineDest(myRank, numDest, numPes, numPatches, numComputes, pComputes, destinations,7);

  startTimer = MPI_Wtime();
  sendTime = 0;
  if (myRank == 0) printf("Doing recvfirst iterations\n");
  for (currIter = 0; currIter < iterations; currIter++){
    sendTime += recvfirst(myRank, numDest, numPes, destinations, msgSize, myComputes, recvbuffer, sendbuffer, req, stat);
    if (myRank == 0 && currIter % 20 == 0){
      printf("k=7 %d iterations elapsed, last 20 iterations took %lf seconds, with a total sendTime of %lf\n", currIter, MPI_Wtime() - startTimer, sendTime);
      startTimer = MPI_Wtime();
      sendTime = 0;
    }
  }

  /*startTimer = MPI_Wtime();
  sendTime = 0;
  if (myRank == 0) printf("Doing sendfirst iterations\n");
  for (currIter = 0; currIter < iterations; currIter++){
    sendTime += sendfirst(myRank, numDest, numPes, destinations, msgSize, myComputes, recvbuffer, sendbuffer, req, stat);
    if (myRank == 0 && currIter % 20 == 0){
      printf("%d iterations elapsed, last 20 iterations took %lf seconds, with a total sendTime of %lf\n", currIter, MPI_Wtime() - startTimer, sendTime);
      startTimer = MPI_Wtime();
      sendTime = 0;
    }
  }*/

  MPI_Finalize();
  return 0;
} /* end function main */

