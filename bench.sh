#!/bin/sh

SERVER=$1
SOURCE=$2
BODY_FILE_NAME=temp_body

./device.py --server $SERVER --source $SOURCE --nowait --genfile $BODY_FILE_NAME

ab -T 'application/octet-stream' -p emulator_data.body -H 'Crayfis-version-code: 1' -H 'Run-id: 2' -H 'Device-id: 21' -H 'Crayfis-version: emulator v0.2' -n 10 -c 2 $SERVER/data.php
