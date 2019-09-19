import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

"""
Corrupt CVBS signal with old-school sync suppression scrambling
"""

# sample rate of input signal in Ms/sec
#samp_rate = 10.5
samp_rate = 14.

# only process this many samples (for quicker testing)
max_samples = 4e6
# process full input signal
#max_samples = None

# numeric values of blank and sync level (varies based on gain used to record signal)
# TODO: determine these automatically
#sync_level = -920
#blank_level = -690
#sync_level = -3880
#blank_level = -2915
sync_level = -1670
blank_level = -1250

def samples():
    # input signal, encoded as little-endian 16-bit ints
    with open('/tmp/cvbs.i16') as f:
        while True:
            buf = f.read(2)
            if not buf:
                return
            yield -struct.unpack('<H', buf)[0]

# TODO track sync len            
def syncs():
    lockout = int(25 * samp_rate * 1000)
    debounce = 10
    sync_thresh = .5*(sync_level + blank_level)
    last_sync = 1e100 # set to infinity so no sync registered until we see an above-sync sample first
    for i, val in enumerate(samples()):
        if i < lockout:
            continue

        if val < sync_thresh:
            if last_sync is None:
                last_sync = i
            elif i - last_sync == debounce:
                yield last_sync/samp_rate
        else:
            last_sync = None

def within(pct, epsilon=.005):
    return 1. - epsilon < pct < 1. + epsilon
            
last_h = None
hsyncs = []
for i, s in enumerate(syncs()):
    if max_samples is not None and s > max_samples:
        break

    if not last_h:
        last_h = s
        continue

    elapsed = s - last_h
    if within(elapsed / 63.555):
        hsyncs.append(s)
        last_h = s
    elif elapsed > 63.555:
        print 'sync lost'
        last_h = s

# length of porches in ms
front_porch = 1.5
back_porch = 9.4  # includes sync pulse
# additional padding of the sync suppress to either side of the HBI
padding = .005*63.555
#padding = .015*63.555

ire = (blank_level - sync_level) / 40.
white_level = blank_level + 100*ire

sync_ix = 0
def calc_sync():
    sync_us = hsyncs[sync_ix]
    return (sync_us - front_porch - padding) * samp_rate, (sync_us + back_porch + padding) * samp_rate
sync_start, sync_end = calc_sync()
x = []
y = []

# standard gated sync suppression scrambling
out1 = open('cvbs-scram.i16', 'w')
# sync suppression with inversion (not sure how accurate this is as it erases the VBI unrecoverably)
out2 = open('cvbs-scram-inv.i16', 'w')

# used to turn scrambling on and off cyclically for comparison: [off_secs, on_secs]
#duty_cycle = [2, 2]
duty_cycle = [0, 2]
def scramble(t):
    t *= 1e-6
    return t % (duty_cycle[0] + duty_cycle[1]) > duty_cycle[0]

for i, val in enumerate(samples()):
    val1 = val
    val2 = val
    val = None
    
    t = i/samp_rate
    #x.append(t)
    if scramble(t):
        if i > sync_start:
            val1 = min(val1 + 80*ire, white_level)
            val2 = min(val2 + 47.5*ire, white_level)
        # invert
        val2 = min(white_level - (val2 - (blank_level + 7.5*ire)), white_level)
            
    #y.append(val)
    out1.write(struct.pack('<H', int(round(-val1))))
    out2.write(struct.pack('<H', int(round(-val2))))
    
    if i > sync_end:
        sync_ix += 1
        if sync_ix == len(hsyncs):
            break
        sync_start, sync_end = calc_sync()

