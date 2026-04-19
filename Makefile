CXX      ?= g++
CXXFLAGS ?= -std=c++98 -Wall -Wextra -pedantic -O2
CPPFLAGS += -Iinclude

BUILD := bin
LIB   := $(BUILD)/libmlp.a
OBJS  := $(patsubst src/%.cpp,$(BUILD)/%.o,$(wildcard src/*.cpp))
BINS  := $(patsubst examples/%.cpp,$(BUILD)/%,$(wildcard examples/*.cpp))

.PHONY: all examples clean

all: $(LIB) examples

examples: $(BINS)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

$(BUILD)/%: examples/%.cpp $(LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LIB) -o $@

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)
