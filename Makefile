MIN=-10
MAX=10
SIZES=100 50000 100000 500000 1000000 5000000 10000000
SCALAR_BENCHMARKS=blackscholes redomap-nontriv
ARRAY_BENCHMARKS=max index-of-max index-of-max-packed mss invred reduce-plus scan-plus
BENCHMARKS=$(SCALAR_BENCHMARKS) $(ARRAY_BENCHMARKS)
EXECUTABLES:=$(BENCHMARKS:%=benchmarks/%-thrust) \
             $(patsubst benchmarks/%-optimised.cu, benchmarks/%-optimised, $(wildcard benchmarks/*-optimised.cu)) \
             $(BENCHMARKS:%=benchmarks/%-futhark) 
# The -I is to use a non-default version of Thrust, for example
# the one from https://github.com/thrust/thrust
NVCFLAGS=-O3 -I $(HOME)/thrust

.PRECIOUS: %-futhark %-thrust %-optimised
.PHONY: benchmark clean build

all: benchmark

# XXX: This runs much faster without -arch=sm_30 for some reason, but
# many of the other benchmarks fail to compile if it's not passed.
benchmarks/mss-thrust: benchmarks/mss-thrust.cu
	nvcc $< -o $@ $(NVCFLAGS)
benchmarks/blackscholes-thrust: benchmarks/blackscholes-thrust.cu
	nvcc $< -o $@ $(NVCFLAGS)
benchmarks/blackscholes-optimised: benchmarks/blackscholes-optimised.cu
	nvcc $< -o $@ $(NVCFLAGS)

%-thrust: %-thrust.cu
	nvcc -arch=sm_30 $< -o $@ $(NVCFLAGS)

%-optimised: %-optimised.cu
	nvcc -arch=sm_30 $< -o $@ $(NVCFLAGS)

$(patsubst benchmarks/%-optimised.cu, benchmark_%, $(wildcard benchmarks/*-optimised.cu)): benchmark_%: benchmarks/%-optimised

%-futhark: %-futhark.fut
	futhark-opencl $< -o $@

$(ARRAY_BENCHMARKS:%=benchmark_%): benchmark_%: $(SIZES:%=data/%_integers) benchmarks/%-thrust benchmarks/%-futhark
	echo; echo; echo "== $*"; \
	tools/run-benchmark.sh $* $(SIZES) 2>>error.log

$(SCALAR_BENCHMARKS:%=benchmark_%): benchmark_%: $(SIZES:%=data/%_scalar) benchmarks/%-thrust benchmarks/%-futhark
	echo; echo; echo "== $*"; \
	tools/run-bench-nontriv.sh $* $(SIZES) 2>>error.log

benchmark: build $(BENCHMARKS:%=benchmark_%)

build: $(SIZES:%=data/%_integers) $(SIZES:%=data/%_scalar) $(EXECUTABLES)

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
