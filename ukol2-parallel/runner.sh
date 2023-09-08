#!/bin/bash

for i in $(find data/ -type f -name "*.mtx") ; do echo $i ; ./main $i 2>/dev/null ; echo "=====================" ; done