default: build


CXX = g++
CXXFLAGS=-O2


build: laplacian.cpp
	$(CXX) $(CXXFLAGS) laplacian.cpp -o laplacian -fopenmp

clean: 
	rm -f *.o *.cuda *.host
