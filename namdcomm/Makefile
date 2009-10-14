#BGPHOME	= /bgsys/drivers/ppcfloor
#CC	= $(BGPHOME)/comm/bin/mpixlcxx
#CFLAGS	= -I$(BGPHOME)/comm/include -g -O3
CC	= mpicxx
CFLAGS = -g

#TRACE_LIB = -L/home/ihchung/hpct_bgp/lib -lmpitrace -lgetarg -llicense
#PREP	= $(HOME)/softwares/scalasca/bin/scalasca -instrument
#SCALIB	= -L$(HOME)/softwares/scalasca/lib/be -lcube3

all: namdcomm 

namdcomm: namdcomm.C
	$(CC) -o namdcomm $< $(CFLAGS)

clean:
	rm -f *.o namdcomm
