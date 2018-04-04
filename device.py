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

    N_CAMERAS = 2

    def __init__(self, server, source_files=None, appcode=None, loc=None, rate=0.5, xb_period=120, fps=30, res=(1920, 1080), gen=None, err=None):
        super().__init__()

        self._server = server

        self._cameras = []
        for i in range(self.N_CAMERAS):
            self._cameras.append(self.Camera(self))

        self._hwid = uuid.uuid1().hex[:16]
        
        self._appcode = appcode
        if not self._appcode:
            alphanums = 'abcdefghijklmnopqrstuvwxyz0123456789'
            alphanums += alphanums.upper()
            alphanums = set(alphanums)
            self._appcode = ''.join(random.sample(alphanums, 7))

        self._loc = loc
        if not self._loc:
            self._loc, placename = random.choice(PLACES)
            print('Using location:{0}'.format(placename))

        self._target_rate = rate
        self._target_res = res
        self._target_fps = fps
        self._xb_period = xb_period
        
        self._room_temp = np.random.normal(loc=230, scale=20)
        self._temp = self._room_temp
        self._plateau_temp_1080p = np.random.normal(loc=350, scale=20) - 230
        self._plateau_temp_pow = np.random.lognormal()

        self._genfile = gen
        self._errfile = err

        self._terminate = threading.Event()


    ''' Utility class for handling multiple cameras in a single device'''
    class Camera:

        RESOLUTIONS = [
                (320, 240),
                (640, 480),
                (1280, 720),
                (1920, 1080), 
                ]

        def __init__(self, dev, source_files=None):

            # keep a reference of the device to access config variables
            self._dev = dev
            self._source = source_files

            if source_files:
                self._event_stream = self._generate_from_source(source_files)
            else:
                self._event_stream = self._simulate_events()

            # randomly determine which higher resolutions are available
            rand = random.random()
            if rand < 0.5:
                # QHD
                self.RESOLUTIONS.append((2560, 1440))
            if rand < 0.25:
                # UHD
                self.RESOLUTIONS.append((3840, 2160))

        ''' construct best fit given allowed fps/res combinations '''
        def _set_stream_cfg(self, target_res, target_fps):
            target_area = target_res[0] * target_res[1]
            self._res = sorted(self.RESOLUTIONS, key=(lambda res: abs(res[0]*res[1] - target_area)))[0]
            self._fps = target_fps
            self._is_calibrated = True
    
        ''' generator to yield source data '''
        def _generate_from_source(self, source_files):
            self._set_stream_cfg(self._dev._target_res, self._dev._target_fps)
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


        ''' create a distribution from a lens-shading map and randomly add hotcells '''
        def _simulate_events(self):
            
            while True:
                self._set_stream_cfg(self._dev._target_res, self._dev._target_fps)
        
                GAIN = 1/20 # ~5 e-/ADC x 4 ADC/pix_val

                # test distributions
                LENS_SHADE_MAPS = [
                        lambda x,y: 1,
                        lambda x,y: 1 + np.exp(-3*((x-3)**2 + (y+2)**2)), # bad spot
                        lambda x,y: 1 + (x**2 + y**2)/250, # radial distr
                        lambda x,y: 1 + np.exp(x/10)/5, # x only
                        lambda x,y: np.cosh(((x-1)**2 + y**2)**0.5 / 12), # off-radial
                        ]

                reverse_res = (self._res[1], self._res[0])
    
                # tail in actual data seems to fall off roughly exponentially
                def electron_cdf(val):
                    return scipy.stats.expon.cdf(val, scale=self._electron_mean)

                if not hasattr(self, '_lens_shade_map'):
                    self._lens_shade_map = random.choice(LENS_SHADE_MAPS)
                if not hasattr(self, '_weights'):
                    self._weights = np.ones(reverse_res)
                if not hasattr(self, '_electron_mean'):
                    self._electron_mean = np.random.lognormal(mean=2.5, sigma=0.3)
                if not hasattr(self, '_hotcells'):
                    n_hotcells = np.random.poisson(lam=10)
                    if n_hotcells:
                        x = np.random.uniform(low=-8, high=8, size=n_hotcells)
                        y = np.random.uniform(low=-4.5, high=4.5, size=n_hotcells)
                        e = np.random.poisson(lam=20*self._electron_mean, size=n_hotcells)
                        freq = np.random.exponential(scale=0.003, size=n_hotcells)
                        self._hotcells = (x, y, e, freq)
                    else:
                        self._hotcells = []


                # aspect ratio of 16:9
                xvals = np.linspace(-8, 8, self._res[0]) 
                yvals = np.linspace(-4.5, 4.5, self._res[1])
            
                # turn the map function into a numpy array with meshgrid
                xx, yy = np.meshgrid(xvals, yvals, sparse=True)
                gain_map = GAIN * (self._lens_shade_map(xx, yy)*np.ones(reverse_res))

                # find corresponding hotcell coordinates
                x_unscaled, y_unscaled, electrons, freq = self._hotcells
                x = ((x_unscaled/16 + 0.5)*self._res[0]).astype(int)
                y = ((y_unscaled/9 + 0.5)*self._res[1]).astype(int)
                print("Hotcells at: ", list(zip(x, y, electrons, freq)))

                hot_electrons = scipy.sparse.csr_matrix((electrons, (y, x)), shape=reverse_res) 
                hot_freq = scipy.sparse.csr_matrix((freq, (y, x)), shape=reverse_res)

                # now find threshold based on lens-shading map
                self._dev._l1thresh = 0
                prob_pass = 1
                target_prob_pass = self._dev._target_rate / self._fps

                while prob_pass > target_prob_pass:
                    self._dev._l1thresh += 1

                    # The effective spatial threshold due to weighting.
                    #
                    # N.B. the +1 is because values need to be strictly 
                    # greater than the L1 thresh to pass but we want this
                    # as a cdf
                    weighted_thresh = np.floor((self._dev._l1thresh+0.5)/self._weights).astype(int) + 1
                
                    # Now convert thresholds to e- counts
                    electron_thresh = (weighted_thresh / gain_map).astype(int)

                    # Get grid of probabilities each pixel will be below threshold
                    # for any given frame, replacing by hotcell probability where
                    # applicable
                    prob_below_thresh = electron_cdf(electron_thresh)
                
                    if self._hotcells:
                        #FIXME: this probably could be more efficient
                        hot_e_array = hot_electrons.toarray()
                        hot_freq_array = hot_freq.toarray()
                        hotcell_cdf = np.where(electron_thresh < hot_e_array, 1-hot_freq_array, 1)
                        prob_below_thresh = np.where(hot_e_array > 0, hotcell_cdf, prob_below_thresh)

                    prob_pass = 1 - np.product(prob_below_thresh)
                    print('L1 = {0}, pass rate = {1}'.format(self._dev._l1thresh, prob_pass))

                self._rate = prob_pass * self._fps

                # now use relative pass probabilites to make a spatial cdf to sample
                prob_above_thresh = 1 - prob_below_thresh
                spatial_pdf = prob_above_thresh / np.sum(prob_above_thresh)
                spatial_cdf = spatial_pdf.cumsum()

                # finally, construct events
                evt = pb.Event()
                evt.gps_lat = self._dev._loc[0]
                evt.gps_lon = self._dev._loc[1]

                while self._is_calibrated:
                    del evt.pixels[:]
                    evt.timestamp = int(1000*time.time())
                    pixels = []
                    pix_coords = []

                    # assume pixels are independent and pass rate is
                    # the same for remainder of frame w/o replacement
                    while not pixels or random.random() < prob_pass:
                        pix = pb.Pixel()
                        coord = (spatial_cdf > random.random()).argmax()
                
                        # make sure no repeats
                        if coord in pix_coords: continue
                        pix_coords.append(coord)
                
                        pix.x = coord % self._res[0]
                        pix.y = coord // self._res[0]

                        if hot_freq[pix.y, pix.x]:
                            n_electrons = hot_electrons[pix.y, pix.x]
                        else:
                            cdf_array = electron_cdf(np.arange(256/GAIN))   
                            # get val from cdf above the L1 threshold
                            n_electrons = (cdf_array > random.uniform(cdf_array[electron_thresh[pix.y, pix.x]], 1)).argmax()

                        pix.val = int(n_electrons * gain_map[pix.y, pix.x])
                        pix.adjusted_val = int(self._weights[pix.y, pix.x] * pix.val)

                        pixels.append(pix)
                
                    #print("pixel generated at ({0},{1}) with value {2}".format(pix.x, pix.y, pix.val))
                    
                    evt.pixels.extend(pixels) 

                    yield evt

        def stream(self, time):
            if not hasattr(self, '_rate'):
                next(self._event_stream)
            n_events = np.random.poisson(time * self._rate)
            print("uploaded {} events...".format(n_events))
            return it.islice(self._event_stream, n_events)



    ''' make a dummy exposure block and fill it with the given events '''
    def _make_xb(self, camera_id, run_id, interval=120):

        camera = self._cameras[camera_id]

        xb = pb.ExposureBlock()
        xb.events.extend(camera.stream(interval))
        xb.run_id = run_id.int & 0xFFFFFFFF
        xb.start_time = int(time.time()*1e3)
        xb.end_time = int(time.time()*1e3 + interval*1e3)
        xb.gps_lat = self._loc[0]
        xb.gps_lon = self._loc[1]
        xb.daq_state = 2
        xb.res_x = camera._res[0]
        xb.res_y = camera._res[1]
        xb.battery_temp = int(self._temp)
        xb.battery_end_temp = int(self._change_temp(camera))
        xb.L1_thresh = self._l1thresh
        xb.L2_thresh = self._l1thresh - 1
        xb.L1_processed = int(camera._fps * interval)
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

    def _change_temp(self, camera):
        # assume plateau temp has a power law relationship with res*fps
        pix_rate = camera._fps * camera._res[0] * camera._res[1]
        pix_rate_std = 1920*1080*30
        plateau_temp = self._room_temp + self._plateau_temp_1080p * (pix_rate/pix_rate_std)**self._plateau_temp_pow
        
        # use exponential model for dT/dt with Gaussian fluctuations
        self._temp += (plateau_temp - self._temp)/5 + np.random.normal(scale=5)
        print("Temperature changed to {}".format(self._temp))
        return self._temp

    def _apply_commands(self, resp):
        recalibrate=False
        if 'set_xb_period' in resp:
            self._xb_period = resp['set_xb_period']
        if 'set_target_resolution' in resp:
            self._target_res = tuple(map(int, resp['set_target_resolution'].split('x')))
            recalibrate=True
        if 'set_target_fps' in resp:
            self._target_fps = resp['set_target_fps']
            recalibrate=True
        if 'cmd_recalibrate' in resp:
            recalibrate=True
        
        if recalibrate:
            for c in self._cameras:
                c._is_calibrated = False

    def run(self):

        run_id = uuid.uuid1()
        camera_id = random.randrange(self.N_CAMERAS)
        xbn = 1

        while not self._terminate.is_set():

            # sleep for the specified period (give or take 5%)
            sleep_time = self._xb_period * np.random.normal(1,0.05)

            # make an xb for this period with the expected number of events
            with EVT_LOCK:
                xb = self._make_xb(camera_id, run_id, sleep_time)
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
            raise NotImplementedError
            data_dirs = os.listdir('data')
            data_dirs.remove('fetch.sh')
            source_path = os.path.join('data', random.choice(data_dirs))
            source_files = glob(os.path.join(source_path, '*.bin.gz'))
            print('Using files from {0}'.format(source_path))

        dev = Device(args.server, source_files=source_files, appcode=args.appcode, xb_period=args.interval, rate=args.rate, gen=args.genfile, err=args.errfile)
        devices.append(dev)
        # PROTIP: stack trace from other threads doesn't appear
        # on docker logs, but debugging can be done by changing
        # 'start' to 'run'
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
