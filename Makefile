MIN=-10
MAX=10
SIZES=100 50000 100000 500000 1000000 5000000 10000000
BENCHMARKS=max index-of-max index-of-max-packed mss invred reduce-plus scan-plus
# The -I is to use a non-default version of Thrust, for example
# the one from https://github.com/thrust/thrust
NVCFLAGS=-O3 -I $(HOME)/thrust

.PRECIOUS: %-futhark %-thrust
.PHONY: benchmark clean

all: benchmark

%-thrust: %-thrust.cu
	nvcc -arch=sm_30 $< -o $@ $(NVCFLAGS)

%-optimised: %-optimised.cu
	nvcc -arch=sm_30 $< -o $@ $(NVCFLAGS)

# This would be much nicer if we had a BENCHMARK_NONTRIV variable
$(patsubst benchmarks/%-optimised.cu, benchmark_%, $(filter-out benchmarks/redomap-nontriv%, $(wildcard benchmarks/*-optimised.cu))): benchmark_%: benchmarks/%-optimised

%-futhark: %-futhark.fut
	futhark-opencl $< -o $@

benchmark_%: $(SIZES:%=data/%_integers) benchmarks/%-thrust benchmarks/%-futhark
	echo; echo; echo "== $*"; \
	tools/run-benchmark.sh $* $(SIZES) 2>>error.log

benchmark_nontriv: $(SIZES:%=data/%_scalar) benchmarks/redomap-nontriv-thrust benchmarks/redomap-nontriv-optimised benchmarks/redomap-nontriv-futhark
	echo; echo; echo "== redomap-nontriv"; \
	tools/run-bench-nontriv.sh "redomap-nontriv" $(SIZES) 2>>error.log

benchmark: $(SIZES:%=data/%_integers) $(BENCHMARKS:%=benchmark_%) benchmark_nontriv

data/%_integers: tools/randomarray
	mkdir -p data && tools/randomarray $(MIN) $(MAX) $* > $@

data/%_scalar: 
	mkdir -p data && echo $* > $@


tools/randomarray: tools/randomarray.c
	gcc -o $@ $< -O3

clean:
	rm -rf data
	rm -f $(BENCHMARKS:%=benchmarks/%-thrust)
	rm -f $(BENCHMARKS:%=benchmarks/%-optimised)
	rm -f $(BENCHMARKS:%=benchmarks/%-futhark)
	rm -f $(BENCHMARKS:%=benchmarks/%-futhark.c)
	rm -f benchmarks/redomap-nontriv-futhark
	rm -f benchmarks/redomap-nontriv-thrust
	rm -f benchmarks/redomap-nontriv-optimised
	rm -f tools/randomarray
	rm -f error.log
