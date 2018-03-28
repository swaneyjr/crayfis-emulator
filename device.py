#!/usr/bin/env python3

import sys, os
import crayon.crayfis_data_pb2 as pb
from glob import glob
import gzip
import numpy as np
import itertools as it
import time
import uuid
import http.client
import random
import threading

EVT_LOCK = threading.Lock()

PLACES = [
        ((0, 0), 'Null Island'),
        ((-27.467917, 153.027778), 'Brisbane'),
        ((46.2, 6.15), 'Geneva'),
        ((45.5, -73.566667), 'Montreal'),
        ((-12.043333, -77.028333), 'Lima'),
        ((6.666667, -1.616667), 'Kumasi'),
        ((51.166667, 71.433333), 'Astana'),
        ((55.75, 37.616667), 'Moscow'),
        ((35.199167, -111.631111), 'Flagstaff'),
        ((43.616667, -116.2), 'Boise'),
        ((37.688889, -97.336111), 'Wichita'),
        ((36.566667, 136.65), 'Kanazawa'),
        ((61.216667, -149.9), 'Anchorage'),
        ]


class Device(threading.Thread):
    def __init__(self, source_files, server, appcode=None, loc=None, rate=0.2, xb_period=120, fps=30, res=(1920, 1080), gen=None, err=None):
        super().__init__()

        self._event_stream = self._generate_events(source_files)
        self._server = server

        self._hwid = uuid.uuid1().hex[:16]
        
        self._appcode = appcode
        if not self._appcode:
            alphanums = 'abcdefghijklmnopqrstuvwxyz0123456788'
            alphanums += alphanums.upper()
            alphanums = set(alphanums)
            self._appcode = ''.join(random.sample(alphanums, 7))

        self._loc = loc
        if not self._loc:
            self._loc, placename = random.choice(PLACES)
            print('Using location:{0}'.format(placename))

        self._rate = rate
        self._xb_period = xb_period
        self._fps = fps
        self._res = res
        self._temp = 250

        self._genfile = gen
        self._errfile = err

        self._terminate = threading.Event()

    ''' generator to yield source data '''
    def _generate_events(self, source_files):
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
    def _make_xb(self, events, run_id, interval=120):
        xb = pb.ExposureBlock()
        xb.events.extend(events)
        xb.run_id = run_id.int & 0xFFFFFFFF
        xb.start_time = int(time.time()*1e3)
        xb.end_time = int(time.time()*1e3 + interval*1e3)
        xb.gps_lat = self._loc[0]
        xb.gps_lon = self._loc[1]
        xb.daq_state = 2
        xb.res_x = self._res[0]
        xb.res_y = self._res[1]
        xb.battery_temp = self._temp
        xb.battery_end_temp = self._change_temp()
        xb.L1_thresh = 10
        xb.L2_thresh = 9
        xb.L1_processed = int(self._fps * interval)
        xb.L2_processed = len(xb.events)
        xb.L1_pass = xb.L2_processed
        xb.L1_skip = 0
        xb.L2_pass = xb.L2_processed
        xb.L2_skip = 0
        xb.aborted = (np.random.random()>0.995)
        return xb

    def _make_header(self, run_id):
        headers = {}
        headers['Content-type'] = "application/octet-stream"
        headers['Crayfis-version'] = 'emulator v0.1'
        headers['Crayfis-version-code'] = '1'
        headers['Device-id'] = self._hwid
        headers['Run-id'] = str(run_id)
        headers['App-code'] = self._appcode
        return headers

    def _change_temp(self):
        return self._temp

    def run(self):

        run_id = uuid.uuid1()
        xbn = 1

        while not self._terminate.is_set():

            # sleep for the specified period (give or take 5%)
            sleep_time = self._xb_period * np.random.normal(1,0.05)

            # make an xb for this period with the expected number of events
            n_events = np.random.poisson(sleep_time * self._rate)
            #print "sending %d events" % n_events
            with EVT_LOCK:
                xb = self._make_xb(it.islice(self._event_stream, n_events), run_id, sleep_time)
            xb.xbn = xbn
            xbn += 1

            dc = pb.DataChunk()
            dc.exposure_blocks.extend([xb])

            conn = http.client.HTTPConnection(self._server)
            headers = self._make_header(run_id)
            body = dc.SerializeToString()
            if self._genfile:
                f = open(self._genfile, 'w')
                f.write(body)
                f.close()
                print("wrote body to file", self._genfile)
                exit(0)
            else:
                conn.request("POST", "/submit", body, headers)
                resp = conn.getresponse()
                if not resp.status in (200, 202):
                    print("got unexpected status code ({0})".format(resp.status))
                    with open(self._errfile, 'w') as errfile:
                        print(resp.read(), file=errfile)
                        print("wrote error to",  self._errfile)
                else:
                    print("uploaded {} events...".format(n_events))
                    print(resp.read())
            print()
            # flush output
            sys.stdout.flush()

            # okay, we've sent the event. now sleep to simulate the interval
            self._terminate.wait(sleep_time)

    def join(self):
        self._terminate.set()
        super().join()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="emulate a crayfis device by generating data and sending it to the server")
    parser.add_argument("--server", required=True, help="the server hostname/address")
    parser.add_argument("--rate", default=0.2, type=float, help="the nominal event rate in Hz")
    parser.add_argument("--interval", default=120, type=float, help="the nominal communication interval in seconds")
    parser.add_argument("--source", default="htc-cosmic", help="the data source to stream from")
    parser.add_argument("--nowait", action='store_true', help="Do not pause before sending the first event.")
    parser.add_argument("--tlimit", type=int, help="Limit the amount of time to send data (in minutes)")
    parser.add_argument("--genfile", help="when set, save request body to file of this name")
    parser.add_argument("--appcode", help="The API key to send with requests.")
    parser.add_argument("--errfile", default="err.html", help="The file to save errors to")
    parser.add_argument("-N", "--ndev", type=int, default=1, help="number of devices to emulate.")
    args = parser.parse_args()

    source_path = os.path.join('data',args.source)
    if not os.path.exists(source_path):
        print("Unknown source:", args.source, file=sys.stderr)
        sys.exit(1)

    source_files = glob(os.path.join(source_path, '*.bin.gz'))
    if not source_files:
        print("Source is empty!", file=sys.stderr)
        sys.exit(1)

    devices = []

    print('spawning {} devices'.format(args.ndev))
    tstart = time.time()
    for i in range(args.ndev):
        dev = Device(source_files, args.server, appcode=args.appcode, xb_period=args.interval, rate=args.rate, gen=args.genfile, err=args.errfile)
        devices.append(dev)
        dev.start()
        if not args.nowait:
            wait_time = np.random.exponential(args.interval/args.ndev)
            wait_time = min(wait_time, args.interval)
            print('waiting {0:.1f} seconds before spawning next device'.format(wait_time))

    try:
        while True:
            if args.tlimit and (time.time() - tstart)/60 > args.tlimit:
                print("time limit exceeded. quitting.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        # user wants to exit.
        print("shutting down threads")
        for dev in devices:
            dev.join()
