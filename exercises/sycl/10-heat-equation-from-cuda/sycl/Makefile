COMMONDIR=../common

CC=clang
CXX=clang++
CFLAGS=-g -O3 -Wall -I$(COMMONDIR)
CXXFLAGS=`mpicxx --showme:compiler` -g -O3 -Wall -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -I$(COMMONDIR)
LDFLAGS=`mpicxx --showme:link` -g -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80
LIBS=-lpng

EXE=heat_sycl
OBJS=main.o core.o core_sycl.o setup.o utilities.o io.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o

all: $(EXE)

core.o: core.cpp heat.h
core_sycl.o: core_sycl.cpp heat.h
utilities.o: utilities.cpp heat.h
setup.o: setup.cpp heat.h
io.o: io.cpp heat.h
main.o: main.cpp heat.h


$(EXE): $(OBJS) $(OBJS_PNG)
	$(CXX) $(LDFLAGS) $(OBJS) $(OBJS_PNG) $(LIBS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) $(OBJS) $(OBJS_PNG) a.out *.png *~
