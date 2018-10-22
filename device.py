#!/usr/bin/env python3

from camera import Camera

import sys, os
import crayon.crayfis_data_pb2 as pb
from glob import glob
import gzip
import numpy as np
import cv2
import random
import time
import math
from base64 import b64decode
import json
import uuid
import http.client
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
            self._cameras.append(Camera(self, source_files))

        self._hwid = uuid.uuid4().hex[:16]
        print("hwid =", self._hwid)
        
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
        self._plateau_temp_pow = np.random.lognormal(-1, 0.5)
        self._overheat_temp = 410

        self._genfile = gen
        self._errfile = err

        self._terminate = threading.Event()

    def _make_run_config(self):
       
        self._run_id = uuid.uuid1()
        rc = pb.RunConfig()

        rc.id_lo = self._run_id.int & 0xFFFFFFFF
        rc.start_time = int(time.time() * 1000)
        rc.crayfis_build = "emulator v0.2"
        rc.hw_params = "build-manufacturer=emulator;build-device=emulator;build-model=emulator;"
        rc.os_params = "emulator"
        rc.camera_id = self._camera_id

        return rc

    def _make_precal_result(self, camera):

        pc = pb.PreCalibrationResult()
        pc.run_id = self._run_id.int % 2**64
        pc.run_id_hi = self._run_id.int // 2**64
        camera._precal_id = uuid.uuid4()
        pc.precal_id = camera._precal_id.int % 2**64
        pc.precal_id_hi = camera._precal_id.int // 2**64
        pc.start_time = int(time.time() * 1000)
        pc.battery_temp = int(self._temp)

        pc.res_x, pc.res_y = camera._res

        # use the same algorithm as the app for weight dimensions
        res_max = 1500
        block_size = math.gcd(*(camera._res))
        sample_step = int((camera._res[0] * camera._res[1] / res_max) ** 0.5)
        while block_size % sample_step != 0:
            sample_step += 1
            if sample_step > block_size:
                raise ValueError

        sample_res = tuple(r // sample_step for r in camera._res)
        weights = np.full(sample_res, 255)
        fmt = '.jpeg'
        params = (cv2.IMWRITE_JPEG_QUALITY, 100)
        retval, encoded = cv2.imencode(fmt, weights, params=params)
        pc.compressed_weights = bytes(encoded)
        pc.compressed_format = fmt

        pc.end_time = int(time.time() * 1000)
        return pc

    ''' make a dummy exposure block and fill it with the given events '''
    def _make_xb(self, camera, interval=120):

        xb = pb.ExposureBlock()
        xb.events.extend(camera.stream(interval))
        xb.run_id = self._run_id.int % 2**64
        xb.run_id_hi = self._run_id.int // 2**64
        xb.precal_id = camera._precal_id.int % 2**64
        xb.precal_id_hi = camera._precal_id.int // 2**64

        xb.start_time_nano = int(time.time()*1e9)
        xb.end_time_nano = int(time.time()*1e9 + interval*1e9)
        xb.start_time = xb.start_time_nano//1000000
        xb.end_time = xb.end_time_nano//1000000
        xb.start_time_ntp = xb.start_time
        xb.end_time_ntp = xb.end_time

        xb.gps_lat = self._loc[0]
        xb.gps_lon = self._loc[1]
        xb.daq_state = 2
        xb.res_x = camera._res[0]
        xb.res_y = camera._res[1]
        xb.battery_temp = int(self._temp)
        xb.battery_end_temp = int(self._change_temp(camera))
        xb.L1_thresh = camera._l1thresh
        xb.L2_thresh = camera._l1thresh - 1
        xb.L1_processed = int(camera._fps * interval)
        xb.L2_processed = len(xb.events)
        xb.L1_pass = xb.L2_processed
        xb.L1_skip = 0
        xb.L2_pass = xb.L2_processed
        xb.L2_skip = 0
        xb.aborted = (xb.battery_end_temp >= self._overheat_temp)
        return xb

    def _make_header(self):
        return {
            'Content-type': "application/octet-stream",
            'Crayfis-version': 'emulator v0.3',
            'Crayfis-version-code': '1',
            'Device-id': self._hwid,
            'Run-id': str(self._run_id),
            'App-code': self._appcode,
            'Camera-id': self._camera_id,
            }

    def _change_temp(self, camera):
        # assume plateau temp has a power law relationship with res*fps
        pix_rate = camera._fps * camera._res[0] * camera._res[1]
        pix_rate_std = 1920*1080*30
        plateau_temp = self._room_temp + self._plateau_temp_1080p * (pix_rate/pix_rate_std)**self._plateau_temp_pow
        
        # use exponential model for dT/dt with Gaussian fluctuations
        self._temp += (plateau_temp - self._temp)/5 + np.random.normal(scale=5)
        return min(self._temp, self._overheat_temp)

    def _apply_commands(self, resp):
        recalibrate=False
        camera = self._cameras[self._camera_id]
        if 'set_weights' in resp:
            cmd = resp['set_weights']
            if camera._res == (cmd['res_x'], cmd['res_y']):
                camera = self._cameras[cmd['camera_id']]
                compressed = np.array(bytearray(b64decode(cmd['weights'])))
                uncompressed = cv2.imdecode(compressed, 0).transpose()
                resized = cv2.resize(uncompressed, camera._res, interpolation=cv2.INTER_CUBIC)
                camera._weights = resized/255
                camera._precal_id = uuid.UUID(hex=cmd['precal_id'])
                recalibrate=True
        if 'set_hotcells' in resp:
            cmd = resp['set_hotcells']
            if camera._res == (cmd['res_x'], cmd['res_y']):
                camera_id = cmd['camera_id']
                hotcells = set(cmd['hotcells'])
                camera._hotcell_mask.update(hotcells)
                camera._precal_id = uuid.UUID(hex=cmd['precal_id'])
                recalibrate=True
        if 'update_precal' in resp:
            cmd = resp['update_precal']
            camera = self._cameras[self._camera_id]
            if camera._precal_id.hex != cmd['precal_id'] \
                    and self._camera_id == cmd['camera_id'] \
                    and camera._res == (cmd['res_x'], cmd['res_y']): 
                camera.get_precal_from_server()
        if 'set_xb_period' in resp:
            self._xb_period = resp['set_xb_period']
        if 'change_data_rate' in resp:
            cmd = resp['change_data_rate']
            camera = self._cameras[self._camera_id]
            if camera._res == (cmd['res_x'], cmd['res_y']) and 'increase' in cmd: 
                try:
                    res_id = camera.RESOLUTIONS.index(camera._res)
                    if cmd['increase'] and res_id < len(camera.RESOLUTIONS)-1:
                        res_id += 1
                        recalibrate=True
                    elif not cmd['increase'] and res_id > 0:
                        res_id -= 1
                        recalibrate=True
                except ValueError:
                    res_id = 0

                self._target_res = camera.RESOLUTIONS[res_id]
        if 'set_target_resolution' in resp:
            self._target_res = tuple(map(int, resp['set_target_resolution'].split('x')))
            recalibrate=True
            # send another precal result
            self._run_id=None
        if 'set_target_fps' in resp:
            self._target_fps = resp['set_target_fps']
            recalibrate=True
        if 'set_battery_overheat_temp' in resp:
            self._overheat_temp = resp['set_battery_overheat_temp']
        if 'cmd_recalibrate' in resp:
            recalibrate=True
        
        if recalibrate:
            for c in self._cameras:
                c._is_calibrated = False

    def run(self):

        self._camera_id = np.random.randint(self.N_CAMERAS)
        camera = self._cameras[self._camera_id]
        xbn = 1

        while not self._terminate.is_set():

            # sleep for the specified period (give or take 5%)
            sleep_time = self._xb_period * np.random.normal(1,0.05)

            dc = pb.DataChunk()

            # create and send run config
            if not hasattr(self, '_run_id'):
                rc = self._make_run_config()
                dc.run_configs.extend([rc])
                camera.stream(0)

            # send precal if necessary after initializing camera
            if camera._send_precal:
                pc = self._make_precal_result(camera)
                dc.precalibration_results.extend([pc])
                camera._send_precal = False

            # make an xb for this period with the expected number of events
            with EVT_LOCK:
                xb = self._make_xb(camera, sleep_time)
            xb.xbn = xbn
            xbn += 1
            dc.exposure_blocks.extend([xb])  
                
            conn = http.client.HTTPConnection(self._server)
            headers = self._make_header()
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
                    print(resp_body)
                    self._apply_commands(resp_body)
            print()
            # flush output
            sys.stdout.flush()

            # okay, we've sent the event. now sleep to simulate the interval
            self._terminate.wait(sleep_time/250)

    def join(self):
        self._terminate.set()
        super().join()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="emulate a crayfis device by generating data and sending it to the server")
    parser.add_argument("--server", required=True, help="the server hostname/address")
    parser.add_argument("--rate", default=0.5, type=float, help="the nominal event rate in Hz")
    parser.add_argument("--interval", default=120, type=float, help="the nominal communication interval in seconds")
    parser.add_argument("--source", help="Data directory to stream events from.  If empty, use simulated data")
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
            print("Using data from source files")
            datafiles = os.listdir(args.source)
            datafiles.remove('fetch.sh')
            datafiles.remove('consolidate.py')
            source_files = list(map(lambda f: os.path.join(args.source, f), datafiles))

        dev = Device(args.server, source_files=source_files, appcode=args.appcode, xb_period=args.interval, rate=args.rate, gen=args.genfile, err=args.errfile)
        devices.append(dev)
        # PROTIP: stack trace from other threads doesn't appear
        # on docker logs, but debugging can be done by changing
        # 'start' to 'run'
        dev.run()
        if not args.nowait:
            wait_time = np.random.exponential(args.interval/args.ndev)
            wait_time = min(wait_time, args.interval)
            print('waiting {0:.1f} seconds before spawning next device'.format(wait_time))
            time.sleep(wait_time)
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
