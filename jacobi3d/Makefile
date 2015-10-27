BGPHOME	= /bgsys/drivers/ppcfloor
CC	= mpicxx
CFLAGS	= -I$(BGPHOME)/comm/include -O3

TRACE_LIB = -L/home/ihchung/hpct_bgp/lib -lmpitrace -lgetarg -llicense
PREP	= $(HOME)/softwares/scalasca/bin/scalasca -instrument
SCALIB	= -L$(HOME)/softwares/scalasca/lib/be -lcube3

all: jacobi3d 14ptStencil

jacobi3d: jacobi3d.C
	$(CC) -o jacobi3d $< $(CFLAGS)

14ptStencil: 14ptStencil.C
	$(CC) -o 14ptStencil $< $(CFLAGS)

trace: jacobi3d.C
	$(CC) -o jacobi3d.trace $< $(CFLAGS) $(TRACE_LIB)

scal: jacobi3d.C
	$(PREP) $(CC) -o jacobi3d.scal $< $(CFLAGS) $(SCALIB)

clean:
	rm -f *.o jacobi3d
