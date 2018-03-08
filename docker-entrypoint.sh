#!/bin/bash

# check if an alternative crayon install is specified:
if [ ! -z "$CRAYON_PATH" ]; then
    echo "Installing custom crayon!"
    pip install --upgrade $CRAYON_PATH
fi

