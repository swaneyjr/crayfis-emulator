#!/usr/bin/env python3

import os, fnmatch
import crayon.crayfis_data_pb2 as pb
from crayon.message import load_messages, get_merged_data

def consolidate(msg_dir, max_xb):

    devices = {}
    for root, dirs, files in os.walk(msg_dir):
        for fname in files:
            if fnmatch.fnmatch(fname, '*.msg'):
                dev_id = fname.split('_')[0]
                fname = os.path.join(root, fname)
                try:
                    if len(devices[dev_id]) < max_xb:
                        devices[dev_id].append(fname)
                except KeyError:
                    devices[dev_id] = [fname]

    for dev_id in devices.keys():
        if len(devices[dev_id]) > 10:
            # write DataChunks to file
            messages = load_messages(devices[dev_id])
            dc = get_merged_data(messages)
            outfile = open(dev_id + '.bin', 'wb')
            outfile.write(dc.SerializeToString())
            outfile.close()
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--msg_dir', required=True)
    parser.add_argument('--max_xb', default=100, type=int)
    args = parser.parse_args()

    consolidate(args.msg_dir, args.max_xb)
