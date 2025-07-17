CXXFLAGS = -std=c++20 -O2 -mavx512f -pthread -lnuma
LDFLAGS = -pthread -lnuma

numa_stress: numa_stress.cc

.PHONY: clean
clean:
	rm -f numa_stress
