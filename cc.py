import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

#samp_rate = 10.5
samp_rate = 14.

#sync_level = -920
#blank_level = -690
#sync_level = -3880
#blank_level = -2915
sync_level = -1670
blank_level = -1265

def samples():
    with open('/tmp/cvbs.i16') as f:
        while True:
            buf = f.read(2)
            if not buf:
                return
            yield -struct.unpack('<H', buf)[0]

lockout = int(25 * samp_rate * 1000)
def syncs():
    debounce = 10
    sync_thresh = .5*(sync_level + blank_level)
    last_sync = 1e100 # set to infinity so no sync registered until we see an above-sync sample first
    for i, val in enumerate(samples()):
        if i < lockout:
            continue

        if val < sync_thresh:
            if last_sync is None:
                last_sync = i
        elif last_sync is not None:            
            duration = i - last_sync
            if duration >= debounce:
                yield last_sync/samp_rate, duration/samp_rate
            last_sync = None

def within(pct, epsilon=.005):
    return 1. - epsilon < pct < 1. + epsilon
            
last_h = None
last_sync = None
hsyncs = []
line = None
for i, s in enumerate(syncs()):
    t, dur = s
    if t > 4e6: # just process beginning to start
        break

    if within(dur / 4.7, .025):
        type = 'h'
    elif within(dur / 2.35, .05):
        type = 'eq'
    elif within(dur / 27.0):
        type = 'v'
    else:
        type = None
    s = t, type
    
    if not last_h:
        last_h = s
        last_sync = s
        continue

    elapsed = t - last_h[0]
    if within(elapsed / 63.555):
        if line is not None:
            line += 1
        if type == 'eq' and last_sync[1] == 'h':
            # TODO verify expected line
            line = 0
        hsyncs.append((t, line))
        last_h = s
    elif elapsed > 63.555:
        print 'sync lost'
        last_h = s
        line = None
    # TODO verify expected even field start
    last_sync = s
        
front_porch = 1.5
back_porch = 9.4  # includes sync pulse

ire = (blank_level - sync_level) / 40.
def _ire(x):
    return blank_level + x*ire
white_level = _ire(100)

hsyncs = [hs for hs in hsyncs if hs[1] in (20, 263+20)]

data_stream = (('\x00'*2)*30)+'\x14\x20\x14\x2e\x10\x50Thanks for watching!\x14\x2f'

sync_ix = 0
def calc_sync():
    global data_stream
    
    sync_us = hsyncs[sync_ix][0]
    line = hsyncs[sync_ix][1]
    field = 1 if line < 262 else 2

    if field == 1:
        # TODO don't split control chars
        data = (data_stream + '\x00\x00')[:2]
        data_stream = data_stream[2:]
    else:
        data = '\x00\x00'

    def byte_to_bits(b):
        bits = map(int, list(reversed(('0'*7)+bin(ord(b))[2:]))[:7])
        return bits + [(sum(bits)+1) % 2]
    bits = [0,0,1] + byte_to_bits(data[0]) + byte_to_bits(data[1])
    print data, bits
    
    return (sync_us + back_porch) * samp_rate, (sync_us + 63.555 - front_porch) * samp_rate, bits
sync_start, sync_end, cc_data = calc_sync()
buf = []

out = open('/tmp/out.i16', 'w')

cc_level = _ire(50)


for i, val in enumerate(samples()):
    if i < lockout:
        continue
    
    t = i/samp_rate

    if i > sync_start:
        clock = (i - sync_start) / ((63.555 - front_porch - back_porch) * samp_rate) * 26
        if clock < 7:
            val = .5*(blank_level+cc_level) + .5*(blank_level-cc_level)*math.cos(clock * 2*math.pi)
        else:
            val = (cc_level if (cc_data+[0])[int(clock)-7] else blank_level)
        buf.append(val)

    out.write(struct.pack('<H', int(round(-val))))
        
    if i > sync_end:
        """
        fig, ax = plt.subplots()
        ax.plot([i/samp_rate for i in xrange(len(buf))], buf)
        ax.set(xlabel='time (s)', ylabel='voltage (mV)',
            title='About as simple as it gets, folks')
        ax.grid()
        print hsyncs[sync_ix][1]
        plt.show()

        bits = [1 if v > blank_level+25*ire else 0 for v in [buf[int(37+27.833*(7+i-.3))] for i in xrange(19)]]
        if bits[:3] != [0,0,1] or sum(bits[3:11]) % 2 != 1 or sum(bits[11:19]) % 2 != 1:
            print 'checksum fail', bits

        for offset in (3, 11):
            val = reduce(lambda a,b: 2*a+b, reversed(bits[offset:offset+7]))
            if val != 0:
                print chr(val), '0x%02x' % val, (('0'*7)+bin(val)[2:])[-8:], hsyncs[sync_ix][1]
        """
        buf = []

        sync_ix += 1
        if sync_ix == len(hsyncs):
            break
        sync_start, sync_end, cc_data = calc_sync()


        
