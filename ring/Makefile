OPTS	= -O3 
CHARMC	= $(HOME)/work/charm-opt/bin/charmc -language ampi $(OPTS)
MPICC	= mpixlc $(OPTS)

all: pingpong ring allreduce

pingpong: pingpong.c
	$(CHARMC) pingpong.c -o pp-ampi
	$(MPICC) pingpong.c -o pp-mpi

ring: ring.c
	$(CHARMC) ring.c -o ring-ampi
	$(MPICC) ring.c -o ring-mpi

allreduce: allreduce.c
	$(CHARMC) allreduce.c -o allr-ampi
	$(MPICC) allreduce.c -o allr-mpi

bigsim:
	$(CHARMC) pingpong.c -o pp-bg
	$(CHARMC) ring.c -o ring-bg
	$(CHARMC) allreduce.c -o allr-bg

proj:
	$(CHARMC) pingpong.c -o pp-ampi.prj -tracemode projections
	$(CHARMC) ring.c -o ring-ampi.prj -tracemode projections
	$(CHARMC) allreduce.c -o allr-ampi.prj -tracemode projections

test: all
	./charmrun +p2 ./pgm 100000 16 +vp2

clean:
	rm -f charmrun conv-host moduleinit* *.o pp-mpi pp-ampi pp-bg \
	pp-ampi.prj ring-mpi ring-ampi ring-bg ring-ampi.prj allr-mpi \
	allr-ampi allr-bg allr-ampi.prj *~

