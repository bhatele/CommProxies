CC	= mpicxx
OPTS	= -O3
LIBS	= -L/home/njain/codes/bgqcounter -lprofiler -L/soft/perftools/bgpm/lib -lbgpm -lrt -lstdc++

INCS	=

all: many2many

many2many: many2many.C
	$(CC) $(OPTS) -o $@ $<

clean:
	rm -f many2many
