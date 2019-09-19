import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image

"""
Decode CVBS signal into sequential video frames, with color, and closed captioning stream
Also some code to inject custom closed captions, but in a non-working state
"""

# sample rate of input signal in Ms/sec
samp_rate = 10.5
#samp_rate = 14.

# only process this many samples (for quicker testing)
max_time = 4e6
# process full input signal
#max_time = None

# numeric values of blank and sync level (varies based on gain used to record signal)
# TODO: determine these automatically
#sync_level = -920
#blank_level = -690
#sync_level = -3880
#blank_level = -2915
#sync_level = -1670
#blank_level = -1265
#sync_level = -7625
#blank_level = -5735
#sync_level = -9590
#blank_level = -7750
sync_level = -9230
blank_level = -7470

def samples():
    #with open('cvbs.i16') as f:
    with open('/tmp/cvbs.i16') as f:
    #with open('/tmp/gdsync2.i16') as f:
        while True:
            buf = f.read(2)
            if not buf:
                return
            yield -struct.unpack('<H', buf)[0]

# ignore first N ms of signal to give it time to stabilize
lockout = int(25 * (samp_rate * 1000))
def syncs():
    debounce = 10
    # TODO need to low-pass filter sync pulse? rarely, high frequency ringing screws up detection
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
    if max_time is not None and t > max_time:
        break

    #print t, dur
    
    if within(dur / 4.7, .15):
        type = 'h'
    elif within(dur / 2.35, .60):
        type = 'eq'
    elif within(dur / 27.0, .02):
        type = 'v'
    else:
        type = None
    s = t, type
    
    if not last_h:
        last_h = s
        last_sync = s
        continue

    elapsed = t - last_h[0]
    #print dur, type, elapsed
    if within(elapsed / 63.555, .15):
        if line is not None:
            line += 1
        if type == 'eq':
            # seems macrovision-tolerant? but better to sync against v-sync pulse?
            if last_sync[1] == 'h':
                if line is not None and line != 525:
                    print 'premature sync'
                line = 0
            elif last_h[1] == 'h':
                if line is not None and line != 263:
                    print 'premature sync'
                line = 263
        print t, line
        hsyncs.append((t, line))
        last_h = s
    elif elapsed > 63.555:
        print 'sync lost', elapsed, t, line
        last_h = s
        line = None
    # TODO verify expected even field start
    last_sync = s
        
# length of various signal segments in ms
front_porch = 1.5
back_porch = 9.4  # includes sync pulse
cburst_start = 4.7 + 0.6
cburst_len = 2.5

ire = (blank_level - sync_level) / 40.
def _ire(x):
    return blank_level + x*ire
white_level = _ire(100)

# filter to just CC lines for quicker eia-608 extraction
#hsyncs = [hs for hs in hsyncs if hs[1] in (20, 262+20)] # 262 or 263?

#### code for injecting custom closed caption stream ####
## note: i think this capability is broken w/o some additional work to get it back

STREAM =    '\x14\x29'
CLEAR_BUF = '\x14\x2e'
START_BUF = '\x14\x20'
DISP_BUF =  '\x14\x2f'
FLASH =     '\x14\x28'

ROW_11 = 0x1040
ROW_12 = 0x1340
ROW_13 = 0x1360
ROW_14 = 0x1440
ROW_15 = 0x1460
ROW = ROW_13
STYLE_DEF =   struct.pack('>H', ROW | 0x0)
STYLE_GREEN = struct.pack('>H', ROW | 0x2)
STYLE_BLUE =  struct.pack('>H', ROW | 0x4)
STYLE_CYAN =  struct.pack('>H', ROW | 0x6)
STYLE_RED =   struct.pack('>H', ROW | 0x8)
STYLE_YEL =   struct.pack('>H', ROW | 0xa)
STYLE_MAG =   struct.pack('>H', ROW | 0xc)

BOX = '\x7f'
MUSIC_NOTE = '\x11\x37'

first = BOX
line = first + MUSIC_NOTE + ' Thanks for watching! ' + MUSIC_NOTE + BOX
# {k: v} -- at timestamp k seconds, stream string v
data_stream = {
    4: (STREAM + CLEAR_BUF + STYLE_DEF + line),
    8: (''
        + STYLE_RED + first
        + STYLE_YEL + first
        + STYLE_GREEN + first
        + STYLE_CYAN + first
        + STYLE_BLUE + first
        + STYLE_MAG + first
    ) * 15 + STYLE_DEF + first,
}

sync_ix = 0
def calc_sync():
    sync_us = hsyncs[sync_ix][0]
    line = hsyncs[sync_ix][1]
    field = 1 if line < 262 else 2

    # get next CC stream bytes to inject
    data = ''
    if field == 1 and data_stream:
        curstrt = min(data_stream.keys())
        if curstrt < sync_us/1e6:
            s = data_stream[curstrt]
            data = s[:2]
            if len(data) == 2 and ord(data[0]) >= 32 and ord(data[1]) < 32:
                # 2nd char is control char; don't split
                data = data[0]
            s = s[len(data):]
            if s:
                data_stream[curstrt] = s
            else:
                del data_stream[curstrt]
    data = (data + '\x00\x00')[:2]
            
    def byte_to_bits(b):
        bits = map(int, list(reversed(('0'*7)+bin(ord(b))[2:]))[:7])
        return bits + [(sum(bits)+1) % 2]
    bits = [0,0,1] + byte_to_bits(data[0]) + byte_to_bits(data[1])
    #print data, bits

    #return (sync_us + back_porch) * samp_rate, (sync_us + 63.555 - front_porch) * samp_rate, bits
    return (sync_us - front_porch) * samp_rate, (sync_us - front_porch + 63.555) * samp_rate, bits

sync_start, sync_end, cc_data = calc_sync()
buf = []

out = open('/tmp/out.i16', 'w')

cc_level = _ire(50)


cburst_freq = 315e6/88
Iref = np.array([math.sin(2*math.pi*cburst_freq*i/(samp_rate*1e6)) for i in xrange(int(63.555*samp_rate*1.05))])
Qref = np.array([math.cos(2*math.pi*cburst_freq*i/(samp_rate*1e6)) for i in xrange(int(63.555*samp_rate*1.05))])

width = int(round(480.*4/3*63.555/52.6))
bitmap = np.zeros([525, width, 3])
bmln = lambda i: ((i*2 if i < 263 else (i-263)*2+1) - 41) % 525
frame = 0
ccstream = open('/tmp/cc.out', 'w')
import csv
ccwriter = csv.DictWriter(ccstream, ['frame', 'channel', 'data'])
ccwriter.writeheader()

import time

passfuncs = {
    'chroma': lambda f: f >= cburst_freq - chroma_bw,
    'luma': lambda f: f < cburst_freq - chroma_bw,
    'qam': lambda f: f < chroma_bw,
}
passkernels = {}

cctext = ''
old_ref_phase = 0
ref_phase = 0

for i, val in enumerate(samples()):
    if i < lockout:
        continue
    
    t = i/samp_rate

    if i > sync_start:
        start = time.time()

        # inject CC stream -- note this uses hard-coded timing constants which have since changed
        #clock = (i - sync_start) / ((63.555 - front_porch - back_porch) * samp_rate) * 26
        #if clock < 7:
        #    val = .5*(blank_level+cc_level) + .5*(blank_level-cc_level)*math.cos(clock * 2*math.pi)
        #else:
        #    val = (cc_level if (cc_data+[0])[int(clock)-7] else blank_level)
        buf.append(val)

    #out.write(struct.pack('<H', int(round(-val))))
 
    if i > sync_end:
        #print 'a', '%.7f' % (time.time() - start)
        
        cburst_rng = (int((cburst_start+front_porch)*samp_rate), int((cburst_start+front_porch+cburst_len)*samp_rate))

        #fig, ax = plt.subplots()

        # force all lines to same length, or else color decoding gets weird phase-drift
        # TODO solve this more elegantly
        while len(buf) < 667:
            buf.append(blank_level)
        buf = np.array(buf)[:667]
        assert len(buf) == 667
        #print 'b', '%.7f' % (time.time() - start)
        
        import numpy
        def bandpass(buf, passtype):
            ifft = np.fft.rfft(buf)

            key = (len(ifft), passtype)
            if key not in passkernels:
                passkernels[key] = np.array([1 if passfuncs[passtype](float(i)/len(ifft)*(1e6*samp_rate/2)) else 0 for i in xrange(len(ifft))])
            kernel = passkernels[key]

            return np.fft.irfft(ifft * kernel)
        
        chroma_bw = 1.3e6
        chroma = bandpass(buf, 'chroma')
        luma = bandpass(buf, 'luma')
        #ax.plot([i/samp_rate for i in xrange(len(chroma))], chroma)
        #ax.plot([i/samp_rate for i in xrange(len(chroma))], luma)

        #print 'c', time.time() - start

        
        isig = 2 * Iref[:len(chroma)] * chroma
        qsig = 2 * Qref[:len(chroma)] * chroma
        #print 'd', time.time() - start
        isig = bandpass(isig, 'qam')
        qsig = bandpass(qsig, 'qam')
        #print 'e', time.time() - start
        mag = (isig**2 + qsig**2)**.5
        phase = np.arctan2(-qsig, isig)
        #import pdb;pdb.set_trace()
        # artifact of line buffer being integer # of samples?
        phase_correction = -0.8 * (len(phase) - 63.55555*samp_rate) * cburst_freq/(samp_rate*1e6) * 2*math.pi
        #print len(phase), phase_correction
        phase = phase + (np.arange(len(phase)) / float(len(phase)) * phase_correction)
        #print 'f', time.time() - start
        old_ref_phase = ref_phase
        ref_phase = np.median(phase[cburst_rng[0]:cburst_rng[1]]) + math.radians(57) #-73)

        #elapsed = hsyncs[sync_ix][0] - hsyncs[sync_ix-1][0]
        #print elapsed, ref_phase % (2*math.pi), (old_ref_phase + elapsed*1e-6*cburst_freq*2*math.pi - ref_phase)%(2*math.pi)
        
        #print 'g', time.time() - start

        #ax.plot([i/samp_rate for i in xrange(len(chroma))], isig)
        #ax.plot([i/samp_rate for i in xrange(len(chroma))], qsig)
        #ax.plot([i/samp_rate for i in xrange(len(chroma))], mag)
        #ax.plot([i/samp_rate for i in xrange(len(chroma))], [100*ph for ph in phase])

        #cb = chroma[cburst_rng[0]:cburst_rng[1]]
        #cbph = phase[cburst_rng[0]:cburst_rng[1]]
        #ax.plot([i/samp_rate for i in xrange(len(cb))], cb)
        #ax.plot([i/samp_rate for i in xrange(len(cb))], cbph)

        #ax.grid()
        #plt.show()

        line = hsyncs[sync_ix][1]
        if line is not None:
            #print line, hsyncs[sync_ix][0] - hsyncs[sync_ix-1][0], ref_phase
            if bmln(line) <= 1:
                img = Image.fromarray(bitmap.astype('uint8'), 'RGB')
                img.save('/tmp/frame%05d.png' % frame)
                print 'wrote frame', frame
                frame += 1
                #print 'h', time.time() - start
        
                
            ix = ((np.arange(width) + .5) / width * len(luma)).astype(int)
            #print 'i', time.time() - start
            ay = luma[ix]
            am = mag[ix]
            ap = phase[ix]
            #print 'j', time.time() - start
            ay = (ay - sync_level) / (blank_level - sync_level) * 40
            ay = np.absolute((ay - 47.5) / 92.5)
            ay = np.clip(ay, 0., 1.)
            am = am / (100. * ire)
            isig = am * np.cos(ap - ref_phase)
            qsig = am * np.sin(ap - ref_phase)
            Imax = .5957
            Qmax = .5226
            isig = np.clip(isig, -Imax, Imax)
            qsig = np.clip(qsig, -Qmax, Qmax)
            R = ay + isig*0.956 + qsig*0.619
            G = ay + isig*-0.272 + qsig*-0.647
            B = ay + isig*-1.106 + qsig*1.703
            #print 'k', time.time() - start
            tobyte = lambda arr: np.clip(arr, 0., 1. - 1e-6) * 256.  # epsilon needed to prevent wraparound
            rgb = np.swapaxes(np.array([tobyte(R), tobyte(G), tobyte(B)]), 0, 1)
            #print 'l', time.time() - start
            bitmap[bmln(line)] = rgb
            #print 'm', time.time() - start

            #bitmap[bmln(line)][640] = {
            #    666: [255, 0, 0],
            #    667: [0, 255, 0],
            #    668: [0, 0, 255],
            #}.get(len(buf), [0, 0, 0])

        # extract CC data
        if line in (20, 262+20): # 262 or 263?
            bitaddr = lambda i: (10.9+14.888+1.986*i)*samp_rate
            bits = [1 if v > blank_level+25*ire else 0 for v in [buf[int(bitaddr(i))] for i in xrange(19)]]
            if not all(b == 0 for b in bits) and (bits[:3] != [0,0,1] or sum(bits[3:11]) % 2 != 1 or sum(bits[11:19]) % 2 != 1):
                print 'checksum fail', bits
                bits = [0]*19
                
            ccchars = []
            for offset in (3, 11):
                val = reduce(lambda a,b: 2*a+b, reversed(bits[offset:offset+7]))
                if val != 0:
                    print chr(val), '0x%02x' % val, (('0'*7)+bin(val)[2:])[-8:], hsyncs[sync_ix]
                    ccchars.append(val)
            if ccchars:
                if 32 <= ccchars[0] <= 127:
                    ccdata = ''.join(map(chr, ccchars))
                else:
                    ccdata = unichr(ccchars[0]*256 + ccchars[1])
                ccwriter.writerow({'frame': frame, 'channel': 1 if line < 262 else 2, 'data': ccdata.encode('utf8')})
                ccstream.flush()
                    

        buf = []

        sync_ix += 1
        if sync_ix == len(hsyncs):
            break
        sync_start, sync_end, cc_data = calc_sync()


        
