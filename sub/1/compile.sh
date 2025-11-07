#!/bin/bash


COMPILER="g++"
SOURCE_FILE="startup_code.cpp" 
EXECUTABLE_NAME="learner" 
FLAGS="-std=c++17" 

if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file not found: $SOURCE_FILE"
    exit 1
fi

echo "Compiling $SOURCE_FILE..."
"$COMPILER" $FLAGS "$SOURCE_FILE" -o "$EXECUTABLE_NAME"