BGPHOME	= /bgsys/drivers/ppcfloor
CC	= mpicxx
CFLAGS	= -I$(BGPHOME)/comm/include -O3

TRACE_LIB = -L/home/ihchung/hpct_bgp/lib -lmpitrace -lgetarg -llicense
PREP	= $(HOME)/softwares/scalasca/bin/scalasca -instrument
SCALIB	= -L$(HOME)/softwares/scalasca/lib/be -lcube3
BGPM	= -L/home/njain/codes/bgqcounter -lprofiler -L/soft/perftools/bgpm/lib -lbgpm -lrt -lstdc++

all: jacobi2d 

jacobi2d: jacobi2d.C
	$(CC) -o jacobi2d $< $(CFLAGS)

trace: jacobi2d.C
	$(CC) -o jacobi2d.trace $< $(CFLAGS) $(TRACE_LIB)

scal: jacobi2d.C
	$(PREP) $(CC) -o jacobi2d.scal $< $(CFLAGS) $(SCALIB)

clean:
	rm -f *.o jacobi2d
