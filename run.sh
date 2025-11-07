#!/bin/bash


EXECUTABLE_NAME="learner"
EXPECTED_ARGS=2

if [ "$#" -ne $EXPECTED_ARGS ]; then
    echo "Error: Invalid number of arguments."
    echo "Usage: $0 <hailfinder_file.bif> <data_file.dat>"
    exit 1
fi

BIF_FILE=$1
DATA_FILE=$2

./"$EXECUTABLE_NAME" "$BIF_FILE" "$DATA_FILE"

