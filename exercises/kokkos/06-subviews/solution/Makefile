default: build

ifndef KOKKOS_PATH
  $(error "Please set KOKKOS_PATH")
endif

# CXX = g++
CXX = hipcc
# CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper

# Variables for the Makefile.kokkos
KOKKOS_DEVICES = "HIP"
# KOKKOS_DEVICES = "Cuda"
# KOKKOS_ARCH = "Zen2"
KOKKOS_ARCH = "AMD_GFX90A"
# KOKKOS_ARCH = "Ampere80"

# Include Makefile.kokkos
include $(KOKKOS_PATH)/Makefile.kokkos

build: $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS) subviews.cpp
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOS_LDFLAGS) subviews.cpp $(KOKKOS_LIBS) -o subviews

clean: kokkos-clean
	rm -fr subviews *.o *.cuda *.host desul *.tmp
