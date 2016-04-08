#!/bin/sh

set -e # Die on error

program=$1
shift
inputsizes="$@"

thrustprog=benchmarks/$program-thrust
optimprog=benchmarks/$program-optimised
futprog=benchmarks/$program-futhark

runs=100

average() {
    awk '{sum += strtonum($0)} END{print sum/NR}'
}

for inputsize in $inputsizes; do
    echo "Input size: $inputsize"
    futhark_input_file=data/${inputsize}_integers

    if ! [ -f "$futhark_input_file" ]; then
        echo "$futhark_input_file does not exist."
        exit 1
    fi

    if ! [ -f "$thrustprog" ]; then
        echo "$thrustprog does not exist."
        exit 1
    fi

    if ! [ -f "$futprog" ]; then
        echo "$futprog does not exist."
        exit 1
    fi

    echo -n "$thrustprog average: "
    ./$thrustprog "$runs" "$inputsize" | grep Runtime | sed -r 's/Runtime: *([0-9]+)us/\1/'

    if [ -f "$optimprog" ]; then
        echo -n "$optimprog average: "
        ./$optimprog "$runs" "$inputsize" | grep Runtime | sed -r 's/Runtime: *([0-9]+)us/\1/'
    fi

    echo -n "$futprog average: "
    (./$futprog < "$futhark_input_file" > /dev/null -r "$runs" -t /dev/fd/8) 8>&1 | average
done
