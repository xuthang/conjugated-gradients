#!/bin/bash

cd ..
f=$(find . -type f -name "*.mtx")
echo $f

for data in $f ; do
	echo $data
	cat $data | head -1
	echo "-----------"
done
