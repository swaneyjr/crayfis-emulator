import crayon.crayfis_data_pb2 as pb
import itertools as it
import numpy as np
from scipy import stats, sparse
import time

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
        rand = np.random.random()
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
        self._rate = self._dev._target_rate
        self._l1thresh = 10

        # pick a random device and start streaming it
        f = open(np.random.choice(source_files), 'rb')
        dc = pb.DataChunk.FromString(f.read())
        while True:
            for xb in dc.exposure_blocks:
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
                return stats.expon.cdf(val, scale=self._electron_mean)

            if not hasattr(self, '_lens_shade_map'):
                self._lens_shade_map = np.random.choice(LENS_SHADE_MAPS)
            if not hasattr(self, '_weights'):
                self._weights = np.ones(reverse_res)

            if not hasattr(self, '_electron_mean'):
                self._electron_mean = np.random.lognormal(mean=2.5, sigma=0.3)
            if not hasattr(self, '_hotcells'):
                self._hotcell_mask = set()
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

            hot_electrons = sparse.csr_matrix((electrons, (y, x)), shape=reverse_res) 
            hot_freq = sparse.csr_matrix((freq, (y, x)), shape=reverse_res)

            # now find threshold based on lens-shading map
            self._l1thresh = 0
            prob_below_thresh_l1 = np.zeros(reverse_res)
            electron_thresh_l1 = np.zeros(reverse_res)
            prob_pass_l1 = 1
            target_prob_pass = self._dev._target_rate / self._fps

            while prob_pass_l1 > target_prob_pass:
                prob_pass_l2 = prob_pass_l1
                self._l1thresh += 1

                # The effective spatial threshold due to weighting.
                #
                # N.B. the +1 is because values need to be strictly 
                # greater than the L1 thresh to pass but we want this
                # as a cdf
                weighted_thresh = np.floor((self._l1thresh+0.5)/self._weights).astype(int) + 1
                
                # Now convert thresholds to e- counts
                electron_thresh_l2 = electron_thresh_l1
                electron_thresh_l1 = (weighted_thresh / gain_map).astype(int)

                # Get grid of probabilities each pixel will be below threshold
                # for any given frame, replacing by hotcell probability where
                # applicable
                prob_below_thresh_l2 = prob_below_thresh_l1
                prob_below_thresh_l1 = electron_cdf(electron_thresh_l1)
                
                if self._hotcells:
                    #FIXME: this probably could be more efficient
                    hot_e_array = hot_electrons.toarray()
                    
                    if self._hotcell_mask:
                        # use fancy indexing to mask
                        hot_e_array[list(zip(*map(lambda xy: (xy//self._res[0], xy%self._res[0]), self._hotcell_mask)))] = 0
                    hot_freq_array = hot_freq.toarray()
                    hotcell_cdf_l1 = np.where(electron_thresh_l1 < hot_e_array, 1-hot_freq_array, 1)
                    hotcell_cdf_l2 = np.where(electron_thresh_l2 < hot_e_array, 1-hot_freq_array, 1)
                    prob_below_thresh_l1 = np.where(hot_e_array > 0, hotcell_cdf_l1, prob_below_thresh_l1)
                    prob_below_thresh_l2 = np.where(hot_e_array > 0, hotcell_cdf_l2, prob_below_thresh_l2)

                prob_pass_l2 = prob_pass_l1
                prob_pass_l1 = 1 - np.product(prob_below_thresh_l1)
                print('L1 = {0}, pass rate = {1}'.format(self._l1thresh, prob_pass_l1))

            self._rate = prob_pass_l1 * self._fps

            # now use relative pass probabilites to make a spatial cdf to sample
            prob_above_thresh_l1 = 1 - prob_below_thresh_l1
            prob_above_thresh_l2 = 1 - prob_below_thresh_l2
            spatial_pdf_l1 = prob_above_thresh_l1 / np.sum(prob_above_thresh_l1)
            spatial_pdf_l2 = prob_above_thresh_l2 / np.sum(prob_above_thresh_l2)
            spatial_cdf_l1 = spatial_pdf_l1.cumsum()
            spatial_cdf_l2 = spatial_pdf_l2.cumsum()

            # finally, construct events
            evt = pb.Event()
            evt.gps_lat = self._dev._loc[0]
            evt.gps_lon = self._dev._loc[1]

            while self._is_calibrated:
                del evt.pixels[:]
                evt.timestamp = int(1000*time.time())
                pixels = []
                pix_coords = []

                # assume pixels are independent, pass rate is the
                # same for remainder of frame w/o replacement, and
                # spatial distribution of L2 passes is similar to
                # that of L1 passes
                while not pixels or np.random.random() < prob_pass_l2:
                    pix = pb.Pixel()
                        
                    if not pixels:
                        spatial_cdf = spatial_cdf_l1
                        electron_thresh = electron_thresh_l1
                    else:
                        spatial_cdf = spatial_cdf_l2
                        electron_thresh = electron_thresh_l2

                    coord = (spatial_cdf > np.random.random()).argmax()
                
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
                        n_electrons = (cdf_array > np.random.uniform(cdf_array[electron_thresh[pix.y, pix.x]], 1)).argmax()

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

