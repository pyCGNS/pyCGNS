
LIBDIRS=-L/home_tools/eucass/tools/lib
INCDIRS=\
-I.. \
-I/home_local/eucass/tools/lib/python2.6/site-packages/numpy/core/include \
-I/home_local/eucass/tools/include/python2.6 
LIBLNK=-lpython2.6 -lhdf5 -lpthread -ldl -lutil 
CFLAGS=-pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv \
-O3 -Wall -Wstrict-prototypes -fPIC

all:
	h5cc $(CFLAGS) $(INCDIRS) -c ../CHLone_l3.c -o CHLone_l3.o
	h5cc $(CFLAGS) $(INCDIRS) -c ../CHLone_error.c -o CHLone_error.o
	h5cc $(CFLAGS) $(INCDIRS) -c ../SIDStoPython.c -o SIDStoPython.o
	h5cc $(CFLAGS) $(INCDIRS) -c EmbeddedMAP.c -o EmbeddedMAP.o
	h5cc $(LIBDIRS) -o EmbeddedMAP EmbeddedMAP.o  SIDStoPython.o CHLone_l3.o  CHLone_error.o $(LIBLNK)	
