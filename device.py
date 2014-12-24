#!/usr/bin/env python

import sys, os
import crayfis_data_pb2 as pb
from glob import glob
import gzip
import numpy as np
import itertools as it
import time
import uuid
import httplib
import random
import threading

''' generator to yield source data '''
def generate_events(source_files):
    while True:
        # pick a random input file and start streaming it
        f = random.choice(source_files)

        dc = pb.DataChunk.FromString(gzip.open(f).read())
        for xb in dc.exposure_blocks:
            #print "actual xb has %d events" % len(xb.events)
            #print "actual xb has interval %d" % ((xb.end_time - xb.start_time)/1e3)
            for evt in xb.events:
                yield evt

''' make a dummy exposure block and fill it with the given events '''
def make_xb(events, run_id, interval=120):
    xb = pb.ExposureBlock()
    xb.events.extend(events)
    xb.run_id = run_id.int & 0xFFFFFFFF
    xb.start_time = int(time.time()*1e3)
    xb.end_time = int(time.time()*1e3 + interval)
    xb.gps_lat = np.random.random()
    xb.gps_lon = np.random.random()
    xb.daq_state = 2
    xb.L1_thresh = 10
    xb.L2_thresh = 9
    xb.L1_processed = int(20*interval) # ~ fps * seconds
    xb.L2_processed = len(xb.events)
    xb.L1_pass = xb.L2_processed
    xb.L1_skip = 0
    xb.L2_pass = xb.L2_processed
    xb.L2_skip = 0
    xb.xbn = 1
    xb.aborted = (np.random.random()>0.995)
    return xb

def make_header(hw_id, run_id, app_code):
    headers = {}
    headers['Content-type'] = "application/octet-stream"
    headers['Crayfis-version'] = 'emulator v0.1'
    headers['Crayfis-version-code'] = '1'
    headers['Device-id'] = hw_id
    headers['Run-id'] = str(run_id)
    headers['App-code'] = app_code
    return headers

def do_sim(event_stream, event_lock, args, terminate):
    if args.hwid == None:
        hwid = uuid.uuid1().hex[:16]

    run_id = uuid.uuid1()

    # before sending the first event, pause for a random fraction
    # of the nominal interval. This helps simulate the fact that
    # phones will start running at random times.
    if not args.nowait:
        pause_time = np.random.rand() * args.interval
        print "pausing %.1fs before generating events..." % (pause_time)
        terminate.wait(pause_time)

        if terminate.is_set():
            print "killed before generating events."
            return

        print "event generation begins."

    xbn = 1
    tstart = time.time()
    i = 0
    while not terminate.is_set():
        i += 1
        if args.nlimit and i>args.nlimit:
            print "upload limit exceeded. quitting."
            break
        # sleep for the specified period (give or take 5%)
        sleep_time = args.interval * np.random.normal(1,0.05)

        # make an xb for this period with the expected number of events
        n_events = np.random.poisson(sleep_time * args.rate)
        #print "sending %d events" % n_events
        with evt_lock:
            xb = make_xb(it.islice(event_stream, n_events), run_id, sleep_time)
        xb.xbn = xbn
        xbn += 1

        dc = pb.DataChunk()
        dc.exposure_blocks.extend([xb])

        conn = httplib.HTTPConnection(args.server)
        headers = make_header(hwid, run_id, args.appcode)
        body = dc.SerializeToString()
        if args.genfile:
            f = open(args.genfile, 'w')
            f.write(body)
            f.close()
            print "wrote body to file %s" % args.genfile
            exit(0)
        else:
            conn.request("POST", "/data.php", body, headers)
            resp = conn.getresponse()
            if not resp.status in (200, 202):
                print "got unexpected status code (%d)" % resp.status
                with open(args.errfile, 'w') as errfile:
                    print >> errfile, resp.read()
                    print "wrote error to %s" % args.errfile
            else:
                print "uploaded %d events..." % n_events
                print resp.read()
        print

        # okay, we've sent the event. now sleep to simulate the interval
        terminate.wait(sleep_time)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="emulate a crayfis device by generating data and sending it to the server")
    parser.add_argument("--server", required=True, help="the server hostname/address")
    parser.add_argument("--rate", default=0.2, type=float, help="the nominal event rate in Hz")
    parser.add_argument("--interval", default=120, type=float, help="the nominal communication interval in seconds")
    parser.add_argument("--source", required=True, help="the data source to stream from")
    parser.add_argument("--hwid", help="the device ID. if none is specified, a random one will be generated.")
    parser.add_argument("--nowait", action='store_true', help="Do not pause before sending the first event.")
    parser.add_argument("--nlimit", type=int, help="Limit the number of times to send data")
    parser.add_argument("--tlimit", type=int, help="Limit the amount of time to send data (in minutes)")
    parser.add_argument("--genfile", help="when set, save request body to file of this name")
    parser.add_argument("--appcode", help="The API key to send with requests.")
    parser.add_argument("--errfile", default="err.html", help="The file to save errors to")
    parser.add_argument("-N", "--ndev", type=int, default=1, help="number of devices to emulate.")
    args = parser.parse_args()

    if args.ndev > 1:
        args.hwid = None

    if not args.appcode:
        alphanums = 'abcdefghijklmnopqrstuvwxyz0123456789'
        alphanums += alphanums.upper()
        alphanums = set(alphanums)
        import random
        args.appcode = ''.join(random.sample(alphanums, 7))

    source_path = os.path.join('data',args.source)
    if not os.path.exists(source_path):
        print >> sys.stderr, "Unknown source:", args.source
        sys.exit(1)

    source_files = glob(os.path.join(source_path, '*.bin.gz'))
    if not source_files:
        print >> sys.stderr, "Source is empty!"
        sys.exit(1)

    event_stream = generate_events(source_files)

    evt_lock = threading.Lock()
    terminate = threading.Event()
    threads = []
    print 'spawning %d threads' % args.ndev
    for i in xrange(args.ndev):
        t = threading.Thread(target=do_sim, args=(event_stream, evt_lock, args, terminate))
        threads.append(t)
        t.start()

    try:
        while True:
            if args.tlimit and (time.time() - tstart)/60 > args.tlimit:
                print "time limit exceeded. quitting."
                break
            time.sleep(1)
    except KeyboardInterrupt:
        # user wants to exit.
        print "shutting down threads"
        terminate.set()

    print 'joining threads'
    for t in threads:
        t.join()
    print 'done. bye!'
