#!/bin/bash

SEQ="ukol1-sequential"
CUDA="ukol2-parallel"

# cd $SEQ
# make clean
# make
# cd ..

# cd $CUDA
# make clean
# make
# cd ..

INSTANCES=$(find data/ -type f -name "*.mtx")

echo data,n,elemNum,iterations,time[ms]
for data in $INSTANCES ; do
    echo -n $(basename $data | cut -d '.' -f 1)
    echo -n ,$(cat $data | head -1 | tr ' ' ',')

    ./$SEQ/main $data > out.txt 2>/dev/null
    echo -n ,$(cat out.txt | head -1 | cut -d ' ' -f 1)
    echo -n ,$(cat out.txt | head -2 | tail -1 | cut -d' ' -f 2 | cut -d'm' -f 1) 
    echo
done
