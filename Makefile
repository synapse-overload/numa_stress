numa_stress:
	g++ -std=c++20 -O2 -mavx512f -pthread numa_stress.cc -lnuma -o numa_stress
