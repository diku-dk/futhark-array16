== Empirical evaluation code for the paper "Design and GPGPU Performance of Futhark's Redomap Construct" submitted to ARRAY'16

To run these benchmarks you must have [the Futhark
compiler](https://github.com/HIPERFIT/futhark) a working CUDA
installation (with `nvcc` in your `$PATH`), a working OpenCL setup,
and a working Thrust installation.  You probably also need a *nix-like
system.  We recommend downloading [Thrust from
Github](https://github.com/thrust/thrust) and placing it in your home
directory.  The build system will look there, and we have found that
the Github version of Thrust performs better than the one that is
preinstalled on most systems.

The benchmark system is built using `make` (sorry).  Ideally, you just
run `make` and it will build and run all benchmarks, printing average
runtimes on stdout.  Of course, many things can go wrong.  You may
need to modify the `Makefile` to fix include paths and the like to fit
your system.

-- Make targets

Apart from the main target that does everything, we also have more
fine-grained targets.

For example, `make build` will only build the benchmark programs and
not run them.  The advantage is that you can run `make build -j` to
build everything in parallel.  It is not advisable to run the actual
benchmarking with `-j`, as this will skew the runtimes.

You can use `make_*benchmark*` to run just *benchmark*.  For example,
`make_mss`.

-- Sample output

This is from running on an NVIDIA Titan Black.

```
$ make


== blackscholes
blackscholes
Input size: 100
benchmarks/blackscholes-thrust average: 409
benchmarks/blackscholes-optimised average: 230
benchmarks/blackscholes-futhark average: 91.88
Input size: 50000
benchmarks/blackscholes-thrust average: 801
benchmarks/blackscholes-optimised average: 299
benchmarks/blackscholes-futhark average: 88.06
Input size: 100000
benchmarks/blackscholes-thrust average: 988
benchmarks/blackscholes-optimised average: 355
benchmarks/blackscholes-futhark average: 72.02
Input size: 500000
benchmarks/blackscholes-thrust average: 1624
benchmarks/blackscholes-optimised average: 623
benchmarks/blackscholes-futhark average: 129.87
Input size: 1000000
benchmarks/blackscholes-thrust average: 1980
benchmarks/blackscholes-optimised average: 975
benchmarks/blackscholes-futhark average: 179.18
Input size: 5000000
benchmarks/blackscholes-thrust average: 4651
benchmarks/blackscholes-optimised average: 3578
benchmarks/blackscholes-futhark average: 636.05
Input size: 10000000
benchmarks/blackscholes-thrust average: 7869
benchmarks/blackscholes-optimised average: 6470
benchmarks/blackscholes-futhark average: 1232.08


== redomap-nontriv
redomap-nontriv
Input size: 100
benchmarks/redomap-nontriv-thrust average: 664
benchmarks/redomap-nontriv-optimised average: 404
benchmarks/redomap-nontriv-futhark average: 151.28
Input size: 50000
benchmarks/redomap-nontriv-thrust average: 1256
benchmarks/redomap-nontriv-optimised average: 637
benchmarks/redomap-nontriv-futhark average: 190.98
Input size: 100000
benchmarks/redomap-nontriv-thrust average: 1604
benchmarks/redomap-nontriv-optimised average: 637
benchmarks/redomap-nontriv-futhark average: 285.33
Input size: 500000
benchmarks/redomap-nontriv-thrust average: 1998
benchmarks/redomap-nontriv-optimised average: 819
benchmarks/redomap-nontriv-futhark average: 746.47
Input size: 1000000
benchmarks/redomap-nontriv-thrust average: 2096
benchmarks/redomap-nontriv-optimised average: 844
benchmarks/redomap-nontriv-futhark average: 806.33
Input size: 5000000
benchmarks/redomap-nontriv-thrust average: 3343
benchmarks/redomap-nontriv-optimised average: 1211
benchmarks/redomap-nontriv-futhark average: 1381.66
Input size: 10000000
benchmarks/redomap-nontriv-thrust average: 4952
benchmarks/redomap-nontriv-optimised average: 1825
benchmarks/redomap-nontriv-futhark average: 2160.76


== max
Input size: 100
benchmarks/max-thrust average: 49
benchmarks/max-futhark average: 101.69
Input size: 50000
benchmarks/max-thrust average: 235
benchmarks/max-futhark average: 68.37
Input size: 100000
benchmarks/max-thrust average: 234
benchmarks/max-futhark average: 69.61
Input size: 500000
benchmarks/max-thrust average: 246
benchmarks/max-futhark average: 69.56
Input size: 1000000
benchmarks/max-thrust average: 255
benchmarks/max-futhark average: 80.3
Input size: 5000000
benchmarks/max-thrust average: 324
benchmarks/max-futhark average: 158.46
Input size: 10000000
benchmarks/max-thrust average: 401
benchmarks/max-futhark average: 252.6


== index-of-max
Input size: 100
benchmarks/index-of-max-thrust average: 81
benchmarks/index-of-max-optimised average: 94
benchmarks/index-of-max-futhark average: 104.62
Input size: 50000
benchmarks/index-of-max-thrust average: 326
benchmarks/index-of-max-optimised average: 303
benchmarks/index-of-max-futhark average: 85.75
Input size: 100000
benchmarks/index-of-max-thrust average: 343
benchmarks/index-of-max-optimised average: 304
benchmarks/index-of-max-futhark average: 87.38
Input size: 500000
benchmarks/index-of-max-thrust average: 500
benchmarks/index-of-max-optimised average: 321
benchmarks/index-of-max-futhark average: 86.56
Input size: 1000000
benchmarks/index-of-max-thrust average: 707
benchmarks/index-of-max-optimised average: 340
benchmarks/index-of-max-futhark average: 94.45
Input size: 5000000
benchmarks/index-of-max-thrust average: 2368
benchmarks/index-of-max-optimised average: 501
benchmarks/index-of-max-futhark average: 192.64
Input size: 10000000
benchmarks/index-of-max-thrust average: 4467
benchmarks/index-of-max-optimised average: 698
benchmarks/index-of-max-futhark average: 309.29


== index-of-max-packed
Input size: 100
benchmarks/index-of-max-packed-thrust average: 50
benchmarks/index-of-max-packed-futhark average: 78.09
Input size: 50000
benchmarks/index-of-max-packed-thrust average: 241
benchmarks/index-of-max-packed-futhark average: 71.02
Input size: 100000
benchmarks/index-of-max-packed-thrust average: 237
benchmarks/index-of-max-packed-futhark average: 69.04
Input size: 500000
benchmarks/index-of-max-packed-thrust average: 253
benchmarks/index-of-max-packed-futhark average: 70.91
Input size: 1000000
benchmarks/index-of-max-packed-thrust average: 260
benchmarks/index-of-max-packed-futhark average: 89.34
Input size: 5000000
benchmarks/index-of-max-packed-thrust average: 326
benchmarks/index-of-max-packed-futhark average: 193.55
Input size: 10000000
benchmarks/index-of-max-packed-thrust average: 408
benchmarks/index-of-max-packed-futhark average: 338.98


== mss
Input size: 100
benchmarks/mss-thrust average: 22
benchmarks/mss-futhark average: 202.08
Input size: 50000
benchmarks/mss-thrust average: 297
benchmarks/mss-futhark average: 182.67
Input size: 100000
benchmarks/mss-thrust average: 319
benchmarks/mss-futhark average: 548.81
Input size: 500000
benchmarks/mss-thrust average: 449
benchmarks/mss-futhark average: 592.69
Input size: 1000000
benchmarks/mss-thrust average: 624
benchmarks/mss-futhark average: 620.01
Input size: 5000000
benchmarks/mss-thrust average: 2054
benchmarks/mss-futhark average: 1007.11
Input size: 10000000
benchmarks/mss-thrust average: 3749
benchmarks/mss-futhark average: 1475.84


== invred
Input size: 100
benchmarks/invred-thrust average: 7571
benchmarks/invred-futhark average: 2173.78
Input size: 50000
benchmarks/invred-thrust average: 56688
benchmarks/invred-futhark average: 2175.93
Input size: 100000
benchmarks/invred-thrust average: 65021
benchmarks/invred-futhark average: 2566.37
Input size: 500000
benchmarks/invred-thrust average: 56920
benchmarks/invred-futhark average: 3091.18
Input size: 1000000
benchmarks/invred-thrust average: 59001
benchmarks/invred-futhark average: 3811.17
Input size: 5000000
benchmarks/invred-thrust average: 70326
benchmarks/invred-futhark average: 8541.62
Input size: 10000000
benchmarks/invred-thrust average: 79156
benchmarks/invred-futhark average: 14379.5


== reduce-plus
Input size: 100
benchmarks/reduce-plus-thrust average: 47
benchmarks/reduce-plus-futhark average: 76.78
Input size: 50000
benchmarks/reduce-plus-thrust average: 234
benchmarks/reduce-plus-futhark average: 68.88
Input size: 100000
benchmarks/reduce-plus-thrust average: 234
benchmarks/reduce-plus-futhark average: 69.86
Input size: 500000
benchmarks/reduce-plus-thrust average: 248
benchmarks/reduce-plus-futhark average: 69.27
Input size: 1000000
benchmarks/reduce-plus-thrust average: 254
benchmarks/reduce-plus-futhark average: 79.87
Input size: 5000000
benchmarks/reduce-plus-thrust average: 322
benchmarks/reduce-plus-futhark average: 159.18
Input size: 10000000
benchmarks/reduce-plus-thrust average: 403
benchmarks/reduce-plus-futhark average: 253.15


== scan-plus
Input size: 100
benchmarks/scan-plus-thrust average: 14
benchmarks/scan-plus-futhark average: 363.05
Input size: 50000
benchmarks/scan-plus-thrust average: 262
benchmarks/scan-plus-futhark average: 369.74
Input size: 100000
benchmarks/scan-plus-thrust average: 264
benchmarks/scan-plus-futhark average: 773.99
Input size: 500000
benchmarks/scan-plus-thrust average: 298
benchmarks/scan-plus-futhark average: 1072.87
Input size: 1000000
benchmarks/scan-plus-thrust average: 334
benchmarks/scan-plus-futhark average: 1193.1
Input size: 5000000
benchmarks/scan-plus-thrust average: 662
benchmarks/scan-plus-futhark average: 1610.59
Input size: 10000000
benchmarks/scan-plus-thrust average: 1059
benchmarks/scan-plus-futhark average: 2842.35
```