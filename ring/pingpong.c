/** \file pingpong.c
 *  Author: Abhinav S Bhatele
 *  Date Created: October 6th, 2010
 *  E-mail: bhatele@illinois.edu
 *
 *  This program measures the ping pong latency between two nodes
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
  int numprocs, myrank, value;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  MPI_Status mstat;
  int i, msg_size;
  double sendTime, recvTime;

  char *send_buf = (char *)malloc(MAX_MSG_SIZE);
  char *recv_buf = (char *)malloc(MAX_MSG_SIZE);

  for(i = 0; i < MAX_MSG_SIZE; i++) {
    recv_buf[i] = send_buf[i] = (char) (i & 0xff);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (msg_size=MIN_MSG_SIZE; msg_size<=MAX_MSG_SIZE; msg_size=(msg_size<<1)) {
    if (myrank == 0) {
      // warm-up
      for (i=0; i<2; i++) {
        MPI_Send(send_buf, msg_size, MPI_CHAR, 1, 999, MPI_COMM_WORLD);
        MPI_Recv(recv_buf, msg_size, MPI_CHAR, 1, 999, MPI_COMM_WORLD, &mstat);
      }

      sendTime = MPI_Wtime();
      // if(myrank == 0) BgPrintf("Start of loop at %f \n");
      for (i=0; i<NUM_MSGS; i++) {
        MPI_Send(send_buf, msg_size, MPI_CHAR, 1, 999, MPI_COMM_WORLD);
        MPI_Recv(recv_buf, msg_size, MPI_CHAR, 1, 999, MPI_COMM_WORLD, &mstat);
      }
      // if(myrank == 0) BgPrintf("End of loop at %f \n");
      recvTime = (MPI_Wtime() - sendTime) / (NUM_MSGS * 2);

      // cool down
      for (i=0; i<2; i++) {
        MPI_Send(send_buf, msg_size, MPI_CHAR, 1, 999, MPI_COMM_WORLD);
        MPI_Recv(recv_buf, msg_size, MPI_CHAR, 1, 999, MPI_COMM_WORLD, &mstat);
      }
    } else {
      // warm-up
      for (i=0; i<2; i++) {
        MPI_Recv(recv_buf, msg_size, MPI_CHAR, 0, 999, MPI_COMM_WORLD, &mstat);
        MPI_Send(send_buf, msg_size, MPI_CHAR, 0, 999, MPI_COMM_WORLD);
      }

      for (i=0; i<NUM_MSGS; i++) {
        MPI_Recv(recv_buf, msg_size, MPI_CHAR, 0, 999, MPI_COMM_WORLD, &mstat);
        MPI_Send(send_buf, msg_size, MPI_CHAR, 0, 999, MPI_COMM_WORLD);
      }

      // cool down
      for (i=0; i<2; i++) {
        MPI_Recv(recv_buf, msg_size, MPI_CHAR, 0, 999, MPI_COMM_WORLD, &mstat);
        MPI_Send(send_buf, msg_size, MPI_CHAR, 0, 999, MPI_COMM_WORLD);
      }
    }
    if(myrank == 0) printf("%d %g\n", msg_size, recvTime);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}

