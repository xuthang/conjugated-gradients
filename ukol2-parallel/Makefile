main: src/main.cu
	nvcc src/main.cu -arch=sm_75 -o main

all: main

run: main
	./main data/x104/x104.mtx 

clean:
	rm -f main