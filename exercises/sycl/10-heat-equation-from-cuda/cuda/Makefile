COMMONDIR=../common

CC=gcc
CXX=mpicxx
NVCC=nvcc
CFLAGS=-g -O3 -Wall -I$(COMMONDIR)
NVCCFLAGS=-g -O3 -arch=sm_80
CXXFLAGS=$(CFLAGS)
LDFLAGS=$(CFLAGS)
LIBS=-lpng -lcudart

EXE=heat.x
OBJS=main.o core.o core_acc.o setup.o utilities.o io.o pngwriter.o

all: $(EXE)

core.o: core.cpp heat.h
core_acc.o: core_acc.cu heat.h
utilities.o: utilities.cpp heat.h
setup.o: setup.cpp heat.h
io.o: io.cpp heat.h
main.o: main.cpp heat.h


$(EXE): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) $(LIBS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: $(COMMONDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) $(OBJS) a.out *.png *~
