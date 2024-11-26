default: build

ifndef KOKKOS_PATH
  $(error "Please set KOKKOS_PATH")
endif

CXX = g++
# CXX = hipcc
# CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper

# Define DEVICES and ARCH before including Makefile.kokkos
KOKKOS_DEVICES = "OpenMP"
# KOKKOS_DEVICES = "HIP"
# KOKKOS_DEVICES = "Cuda"
KOKKOS_ARCH = "Zen2"
# KOKKOS_ARCH = "AMD_GFX90A"
# KOKKOS_ARCH = "Ampere80"

# Include Makefile.kokkos
include $(KOKKOS_PATH)/Makefile.kokkos

#build: $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS) hello.cpp
#	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOS_LDFLAGS) hello.cpp $(KOKKOS_LIBS) -o hello
build: $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS) foo.cpp
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOS_LDFLAGS) foo.cpp $(KOKKOS_LIBS) -o hello

clean: kokkos-clean
	rm -f *.o *.cuda *.host
