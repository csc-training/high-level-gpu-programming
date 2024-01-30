COMMONDIR=../common

CXX=mpicxx
CC=gcc
NVCC=nvcc
NVCCFLAGS=-g -O3 -I$(COMMONDIR)
CCFLAGS=-g -O3 -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng -lcudart

EXE=heat_cuda
OBJS=main.o core.o core_cuda.o setup.o utilities.o io.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o

all: $(EXE)

core.o: core.cpp heat.h
core_cuda.o: core_cuda.cu heat.h
utilities.o: utilities.cpp heat.h
setup.o: setup.cpp heat.h
io.o: io.cpp heat.h
main.o: main.cpp heat.h


$(EXE): $(OBJS) $(OBJS_PNG)
	$(CXX) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) $(OBJS) $(OBJS_PNG) a.out *.png *~
