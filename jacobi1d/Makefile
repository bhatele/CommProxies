CC=mpicxx -O3
OPTS=
LIBS=
		 
INCS= 

all:jacobi1d

jacobi1d: jacobi1d.C
	$(CC) $(OPTS) $(INCS) $(LIBS) -o $@ $<

clean:
	rm -f jacobi1d
