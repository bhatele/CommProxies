OPTS	= -O3 
CHARMC	= $(HOME)/work/charm/bin/charmc -language ampi $(OPTS)
MPICC	= mpixlc $(OPTS)

all: ring allreduce

ring: ring.c
	$(CHARMC) ring.c -o ring-ampi
	$(MPICC) ring.c -o ring-mpi

allreduce: allreduce.c
	$(CHARMC) allreduce.c -o allr-ampi
	$(MPICC) allreduce.c -o allr-mpi

bigsim:
	$(CHARMC) allreduce.c -o allr-bg
	$(CHARMC) ring.c -o ring-bg

test: all
	./charmrun +p2 ./pgm 100000 16 +vp2

clean:
	rm -f charmrun conv-host moduleinit* *.o ring-* allr-* *~

