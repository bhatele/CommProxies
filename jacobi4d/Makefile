BGPHOME	= /bgsys/drivers/ppcfloor
CC	= mpicxx
CFLAGS	= -I$(BGPHOME)/comm/include -O3

TRACE_LIB = -L/home/ihchung/hpct_bgp/lib -lmpitrace -lgetarg -llicense
PREP	= $(HOME)/softwares/scalasca/bin/scalasca -instrument
SCALIB	= -L$(HOME)/softwares/scalasca/lib/be -lcube3

all: jacobi4d 

jacobi4d: jacobi4d.C
	$(CC) -o jacobi4d $< $(CFLAGS)

trace: jacobi4d.C
	$(CC) -o jacobi4d.trace $< $(CFLAGS) $(TRACE_LIB)

scal: jacobi4d.C
	$(PREP) $(CC) -o jacobi4d.scal $< $(CFLAGS) $(SCALIB)

clean:
	rm -f *.o jacobi4d
