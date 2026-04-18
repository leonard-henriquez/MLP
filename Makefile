CXX      ?= g++
CXXFLAGS ?= -std=c++98 -Wall -Wextra -pedantic -O2
CPPFLAGS += -Iinclude

BUILD := bin
LIB   := $(BUILD)/libmlp.a
OBJS  := $(patsubst src/%.cpp,$(BUILD)/%.o,$(wildcard src/*.cpp))

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)
