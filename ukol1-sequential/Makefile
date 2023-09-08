#CXX = g++
CFLAGS =-Wall -pedantic -O2 -g -std=c++17
#LDFLAGS =-lcrypto
LDFLAGS =

#change compilation name here
PROGRAM_NAME = main

#important, use . if for current directory, dont leave empty
SOURCE = src
BUILD = .

all: $(PROGRAM_NAME)

run: $(PROGRAM_NAME)
	./$<

debug: $(PROGRAM_NAME)
	valgrind --leak-check=full ./$<

$(PROGRAM_NAME): $(BUILD)/$(PROGRAM_NAME).o
	$(CXX) -o $@ $^ $(LDFLAGS)

$(BUILD)/%.o: $(SOURCE)/%.cpp | $(BUILD)
	$(CXX) $(CFLAGS) -c -o $@ $<

$(BUILD):
	mkdir -p $@

.PHONY : clean
clean :
	-rm $(BUILD)/*.o $(PROGRAM_NAME)

#$< 1st prereq
#$^ all prereq
#$@ var before ":"
