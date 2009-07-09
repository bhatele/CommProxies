OPTS	= -O3 
CHARMC	= $(HOME)/work/opt-charm/charm/bin/charmc -language ampi $(OPTS)
MPICC	= mpixlc $(OPTS)

all: bgsim mpi

bgsim: ring.c
	$(CHARMC) ring.c -o ring-ampi

mpi: ring.c
	$(MPICC) ring.c -o ring-mpi

test: all
	./charmrun +p2 ./pgm 100000 16 +vp2

clean:
	rm -rf charmrun conv-host moduleinit* *.o ring ring-mpi *~ *.sts

