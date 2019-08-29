import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

samp_rate = 10.5
#samp_rate = 14.

#sync_level = -920
#blank_level = -690
#sync_level = -3880
#blank_level = -2915
#sync_level = -1670
#blank_level = -1265
#sync_level = -7625
#blank_level = -5735
sync_level = -9590
blank_level = -7750

def samples():
    with open('gladiator.i16') as f:
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
    if t > 0.5e6: # just process beginning to start
        break

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
        if type == 'eq' and last_sync[1] == 'h':
            # TODO verify expected line
            line = 0
        print t, line
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
cburst_start = 4.7 + 0.6
cburst_len = 2.5

ire = (blank_level - sync_level) / 40.
def _ire(x):
    return blank_level + x*ire
white_level = _ire(100)

#hsyncs = [hs for hs in hsyncs if hs[1] in (20, 262+20)] # 262 or 263?

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


bitmap = open('/tmp/bitmap.u8', 'w')

cburst_freq = 315e6/88
Iref = [math.sin(2*math.pi*cburst_freq*i/(samp_rate*1e6)) for i in xrange(int(63.555*samp_rate*1.05))]
Qref = [math.cos(2*math.pi*cburst_freq*i/(samp_rate*1e6)) for i in xrange(int(63.555*samp_rate*1.05))]

for i, val in enumerate(samples()):
    if i < lockout:
        continue
    
    t = i/samp_rate

    if i > sync_start:
        #clock = (i - sync_start) / ((63.555 - front_porch - back_porch) * samp_rate) * 26
        #if clock < 7:
        #    val = .5*(blank_level+cc_level) + .5*(blank_level-cc_level)*math.cos(clock * 2*math.pi)
        #else:
        #    val = (cc_level if (cc_data+[0])[int(clock)-7] else blank_level)
        buf.append(val)

    #out.write(struct.pack('<H', int(round(-val))))
 
    if i > sync_end:
        cburst_rng = (int((cburst_start+front_porch)*samp_rate), int((cburst_start+front_porch+cburst_len)*samp_rate))

        #fig, ax = plt.subplots()

        import numpy
        def bandpass(buf, passfunc):
            ifft = numpy.fft.rfft(buf)
            ifft = [n if passfunc(float(i)/len(ifft)*(1e6*samp_rate/2)) else 0 for i, n in enumerate(ifft)]
            return numpy.fft.irfft(ifft)
        
        chroma_bw = 1.5e6
        chroma = bandpass(buf, lambda f: f >= cburst_freq - chroma_bw)
        luma = [a-b for a, b in zip(buf,chroma)]
        #ax.plot([i/samp_rate for i in xrange(len(chroma))], chroma)
        #ax.plot([i/samp_rate for i in xrange(len(chroma))], luma)

        isig = [2*Iref[k]*chroma[k] for k in xrange(len(chroma))]
        qsig = [2*Qref[k]*chroma[k] for k in xrange(len(chroma))]
        isig = bandpass(isig, lambda f: f < chroma_bw)
        qsig = bandpass(qsig, lambda f: f < chroma_bw)
        mag = [(a**2+b**2)**.5 for a, b in zip(isig, qsig)]
        phase = [math.atan2(b, a) for a, b in zip(isig, qsig)]
        cburst_phase = phase[cburst_rng[0]:cburst_rng[1]]
        ref_phase = list(sorted(cburst_phase))[len(cburst_phase)/2]
        isig = [m*math.cos(theta - ref_phase) for m, theta in zip(mag, phase)]
        qsig = [m*math.sin(theta - ref_phase) for m, theta in zip(mag, phase)]

        #ax.plot([i/samp_rate for i in xrange(len(chroma))], isig)
        #ax.plot([i/samp_rate for i in xrange(len(chroma))], qsig)
        #ax.plot([i/samp_rate for i in xrange(len(chroma))], mag)
        #ax.plot([i/samp_rate for i in xrange(len(chroma))], phase)

        #cb = chroma[cburst_rng[0]:cburst_rng[1]]
        #cbph = phase[cburst_rng[0]:cburst_rng[1]]
        #ax.plot([i/samp_rate for i in xrange(len(cb))], cb)
        #ax.plot([i/samp_rate for i in xrange(len(cb))], cbph)
        
        #ax.grid()
        #plt.show()
        
        width = int(round(480.*4/3*63.555/52.6))
        for c in xrange(width):
            ix = int((c+.5)/width*len(luma))
            y = luma[ix]
            #y = (float(y) - sync_level) / (blank_level - sync_level) / (140 / 40.)
            
            y = (float(y) - sync_level) / (blank_level - sync_level) * 40
            y = abs((y - 47.5) / 92.5)
            #print luma[ix], y

            y = max(min(y, 1. - 1e-3), 0.)
            cI = isig[ix] / (100. * ire)
            cQ = qsig[ix] / (100. * ire)
            #print 'yiq', y, cI, cQ
            R = y + cI*0.956 + cQ*0.619
            G = y + cI*-0.272 + cQ*-0.647
            B = y + cI*-1.106 + cQ*1.703
            #print 'rgb', R, G, B
            bitmap.write(chr(max(min(int(256.*R),255),0)))
            bitmap.write(chr(max(min(int(256.*G),255),0)))
            bitmap.write(chr(max(min(int(256.*B),255),0)))
        
        if hsyncs[sync_ix][1] in (20, 262+20): # 262 or 263?
            #bitaddr = lambda i: (37+27.833*(7+i-.3)) / 14. * samp_rate
            bitaddr = lambda i: (14.888+1.986*i)*samp_rate
            bits = [1 if v > blank_level+25*ire else 0 for v in [buf[int(bitaddr(i))] for i in xrange(19)]]
            if not all(b == 0 for b in bits) and (bits[:3] != [0,0,1] or sum(bits[3:11]) % 2 != 1 or sum(bits[11:19]) % 2 != 1):
                print 'checksum fail', bits

            for offset in (3, 11):
                val = reduce(lambda a,b: 2*a+b, reversed(bits[offset:offset+7]))
                if val != 0:
                    print chr(val), '0x%02x' % val, (('0'*7)+bin(val)[2:])[-8:], hsyncs[sync_ix][1]

        buf = []

        sync_ix += 1
        if sync_ix == len(hsyncs):
            break
        sync_start, sync_end, cc_data = calc_sync()


        
