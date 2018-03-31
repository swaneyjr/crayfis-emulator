#!/usr/bin/env python3

import sys, os
import crayon.crayfis_data_pb2 as pb
from glob import glob
import gzip
import numpy as np
import scipy.stats
import itertools as it
import time
import json
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

    RESOLUTIONS = [
            (320, 240),
            (640, 480), 
            (1280, 720), 
            (1920, 1080),
            ]

    def __init__(self, server, source_files=None, appcode=None, loc=None, rate=0.5, xb_period=120, fps=30, res=(1920, 1080), gen=None, err=None):
        super().__init__()

        if source_files:
            self._event_stream = self._generate_from_source(source_files)
        else:
            self._event_stream = self._simulate_events()

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

        self._target_rate = rate
        self._xb_period = xb_period
        self._fps = fps

        # randomly determine which higher resolutions are available
        rand = random.random()
        if rand < 0.5:
            # QHD
            self.RESOLUTIONS.append((2560, 1440))
        if rand < 0.25:
            # UHD
            self.RESOLUTIONS.append((3840, 2160))

        self._set_res(res)
        
        self._room_temp = np.random.normal(loc=230, scale=20)
        self._temp = self._room_temp
        self._plateau_temp_1080p = np.random.normal(loc=350, scale=20) - 230
        self._plateau_temp_pow = np.random.lognormal()

        self._genfile = gen
        self._errfile = err

        self._terminate = threading.Event()

        # run once to configure variables
        next(self._event_stream)


    ''' generator to yield source data '''
    def _generate_from_source(self, source_files):
        self._rate = self._target_rate
        self._l1thresh = 10
        while True:
            # pick a random input file and start streaming it
            f = random.choice(source_files)

            dc = pb.DataChunk.FromString(gzip.open(f).read())
            for xb in dc.exposure_blocks:
                #print "actual xb has %d events" % len(xb.events)
                #print "actual xb has interval %d" % ((xb.end_time - xb.start_time)/1e3)
                for evt in xb.events:
                    yield evt

    def _simulate_events(self):
        
        # test distributions
        LENS_SHADE_MAPS = [
                lambda x,y: 1,
                lambda x,y: 1 + np.exp(-3*((x-3)**2 + (y+2)**2)), # bad spot
                lambda x,y: 1 + (x**2 + y**2)/250, # radial distr
                lambda x,y: 1 + np.exp(x/10)/5, # x only
                lambda x,y: np.cosh(((x-1)**2 + y**2)**0.5 / 12), # off-radial
                ]

        # tail in actual data seems to fall off roughly exponentially
        PIXVAL_CDF = lambda val: scipy.stats.expon.cdf(val, scale=self._pixval_mean)

        if not hasattr(self, '_lens_shade_map'):
            self._lens_shade_map = random.choice(LENS_SHADE_MAPS)
        if not hasattr(self, '_weights'):
            self._weights = np.ones(self._res)
        if not hasattr(self, '_pixval_mean'):
            self._pixval_mean = np.random.lognormal(mean=-0.5, sigma=0.5)

        # aspect ratio of 16:9
        xvals = np.linspace(-8, 8, self._res[0]) 
        yvals = np.linspace(-4.5, 4.5, self._res[1])
            
        # turn the map function into a numpy array with meshgrid
        xx, yy = np.meshgrid(xvals, yvals, sparse=True)
        lens_shade_grid = (self._lens_shade_map(xx, yy) * np.ones((self._res[1], self._res[0]))).T


        # now find threshold based on lens-shading map
        self._l1thresh = 0
        prob_pass = 1
        target_prob_pass = self._target_rate / self._fps

        while prob_pass > target_prob_pass:
            self._l1thresh += 1

            # The effective spatial threshold due to weighting.
            #
            # N.B. the +1 is because values need to be strictly 
            # greater than the L1 thresh to pass but we want this
            # as a cdf
            weighted_thresh = np.floor((self._l1thresh+0.5)/self._weights) + 1 
                
            # Now undo the effects of weighting
            thresh_before_shading = weighted_thresh / lens_shade_grid

            # Get grid of probabilities each pixel will be below threshold
            # for any given frame
            prob_below_thresh = PIXVAL_CDF(thresh_before_shading)

            prob_pass = 1 - np.product(prob_below_thresh)
            print('L1 = {0}, pass rate = {1}'.format(self._l1thresh, prob_pass))

        self._rate = prob_pass * self._fps

        # now use relative pass probabilites to make a spatial cdf to sample
        prob_above_thresh = 1 - prob_below_thresh
        spatial_pdf = prob_above_thresh / np.sum(prob_above_thresh)
        spatial_cdf = spatial_pdf.cumsum()

        # finally, construct events
        evt = pb.Event()
        evt.gps_lat = self._loc[0]
        evt.gps_lon = self._loc[1]

        while True:
            del evt.pixels[:]
            evt.timestamp = int(1000*time.time())
            pixels = []
            pix_coords = []

            # assume pixels are independent and pass rate is
            # the same for remainder of frame w/o replacement
            while not pixels: #or random.random() < prob_pass:
                pix = pb.Pixel()
                coord = (spatial_cdf > random.random()).argmax()
                
                # make sure no repeats
                if coord in pix_coords: continue
                pix_coords.append(coord)
                
                pix.x = coord % self._res[0]
                pix.y = coord // self._res[0]

                lens_shade_coeff = lens_shade_grid[pix.x, pix.y]
                val_cdf = 1 - np.exp(-self._pixval_mean * lens_shade_coeff * np.arange(256))

                pix.val = (val_cdf > random.uniform(val_cdf[self._l1thresh], 1)).argmax()
                pix.adjusted_val = int(self._weights[pix.x, pix.y] * pix.val)

                pixels.append(pix)
                
            print("pixel generated at ({0},{1}) with value {2}".format(pix.x, pix.y, pix.val))
                
            evt.pixels.extend(pixels) 

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
        xb.battery_temp = int(self._temp)
        xb.battery_end_temp = int(self._change_temp())
        xb.L1_thresh = self._l1thresh
        xb.L2_thresh = self._l1thresh - 1
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
        # assume plateau temp has a power law relationship with res*fps
        pix_rate = self._fps * self._res[0] * self._res[1]
        pix_rate_std = 1920*1080*30
        plateau_temp = self._room_temp + self._plateau_temp_1080p * (pix_rate/pix_rate_std)**self._plateau_temp_pow
        
        # use exponential model for dT/dt with Gaussian fluctuations
        self._temp += (plateau_temp - self._temp)/5 + np.random.normal(scale=5)
        print("Temperature changed to {}".format(self._temp))
        return self._temp

    def _apply_commands(self, resp):
        if 'set_xb_period' in resp:
            self._xb_period = resp['set_xb_period']
        if 'set_target_resolution' in resp:
            target_res = tuple(map(int, resp['set_target_resolution'].split('x')))
            self._set_res(target_res)
        if 'set_target_fps' in resp:
            self._fps = resp['set_target_fps']

    def _set_res(self, request):
        request_area = request[0] * request[1]
        self._res = sorted(self.RESOLUTIONS, key=(lambda res: abs(res[0]*res[1] - request_area)))[0]

    def run(self):

        run_id = uuid.uuid1()
        xbn = 1

        while not self._terminate.is_set():

            # sleep for the specified period (give or take 5%)
            sleep_time = self._xb_period * np.random.normal(1,0.05)

            # make an xb for this period with the expected number of events
            n_events = np.random.poisson(sleep_time * self._rate)
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
                    resp_body = json.loads(resp.read())
                    self._apply_commands(resp_body)
                    print(resp_body)
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
    parser.add_argument("--rate", default=0.5, type=float, help="the nominal event rate in Hz")
    parser.add_argument("--interval", default=120, type=float, help="the nominal communication interval in seconds")
    parser.add_argument("--source", action='store_true', help="Stream events from live data")
    parser.add_argument("--nowait", action='store_true', help="Do not pause before sending the first event.")
    parser.add_argument("--tlimit", type=int, help="Limit the amount of time to send data (in minutes)")
    parser.add_argument("--genfile", help="when set, save request body to file of this name")
    parser.add_argument("--appcode", help="The API key to send with requests.")
    parser.add_argument("--errfile", default="err.html", help="The file to save errors to")
    parser.add_argument("-N", "--ndev", type=int, default=1, help="number of devices to emulate.")
    args = parser.parse_args()

    print('spawning {} devices'.format(args.ndev))

    devices = []
    tstart = time.time()

    for i in range(args.ndev):
        
        # pick a random file if the source was never specified
        source_files=None
        if args.source:
            data_dirs = os.listdir('data')
            data_dirs.remove('fetch.sh')
            source_path = os.path.join('data', random.choice(data_dirs))
            source_files = glob(os.path.join(source_path, '*.bin.gz'))
            print('Using files from {0}'.format(source_path))

        dev = Device(args.server, source_files=source_files, appcode=args.appcode, xb_period=args.interval, rate=args.rate, gen=args.genfile, err=args.errfile)
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
