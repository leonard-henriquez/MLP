CXX      ?= g++
CXXFLAGS ?= -std=c++98 -Wall -Wextra -pedantic -O2
CPPFLAGS += -Iinclude

BUILD := bin
LIB   := $(BUILD)/libmlp.a
OBJS  := $(patsubst src/%.cpp,$(BUILD)/%.o,$(wildcard src/*.cpp))
BINS  := $(patsubst examples/%.cpp,$(BUILD)/%,$(wildcard examples/*.cpp)) $(BUILD)/tui
TUI   := $(wildcard examples/tui/*.cpp)
TESTS := $(patsubst tests/%.cpp,$(BUILD)/tests/%.o,$(wildcard tests/*.cpp))

.PHONY: all examples test clean

all: $(LIB) examples

examples: $(BINS)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

test: $(BUILD)/run_tests
	./$(BUILD)/run_tests

$(BUILD)/run_tests: $(TESTS) $(LIB)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/tests/%.o: tests/%.cpp tests/check.h | $(BUILD)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%: examples/%.cpp $(LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LIB) -o $@

$(BUILD)/tui: $(TUI) $(wildcard examples/tui/*.h) $(LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TUI) $(LIB) -o $@

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)
