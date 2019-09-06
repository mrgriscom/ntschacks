import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import bisect
import time

samp_rate = 640/52.6

colors = {
    'grey':    (77,   0, 0),
    'yellow':  (69, 167, 56.5),
    'cyan':    (56, 283, 56.5),
    'green':   (48, 241, 56.5),
    'magenta': (36,  61, 56.5),
    'red':     (28, 103, 56.5),
    'blue':    (15, 347, 56.5),
    'white':  (100, 0, 0),
    'black':  (7.5,   0, 0),
    '-I':     (7.5, 303, 20),
    '+Q':     (7.5,  33, 20),
    'pluge+': (11.5, 0, 0),
    'pluge-': (3.5, 0, 0),
}

rows = [
    (2/3., [
        (1/7., 'grey'),
        (1/7., 'yellow'),
        (1/7., 'cyan'),
        (1/7., 'green'),
        (1/7., 'magenta'),
        (1/7., 'red'),
        (1/7., 'blue'),
    ]),
    (1/12., [
        (1/7., 'blue'),
        (1/7., 'black'),
        (1/7., 'magenta'),
        (1/7., 'black'),
        (1/7., 'cyan'),
        (1/7., 'black'),
        (1/7., 'grey'),
    ]),
    (1/4., [
        (5/28., '-I'),
        (5/28., 'white'),
        (5/28., '+Q'),
        (5/28., 'black'),
        (1/21., 'pluge-'),
        (1/21., 'black'),
        (1/21., 'pluge+'),
        (1/7., 'black'),
    ]),
]


cburst_freq = 315e6/88
Iref = np.array([math.sin(2*math.pi*cburst_freq*i/(samp_rate*1e6)) for i in xrange(int(63.555*samp_rate*1.05))])
Qref = np.array([math.cos(2*math.pi*cburst_freq*i/(samp_rate*1e6)) for i in xrange(int(63.555*samp_rate*1.05))])

width = 640
height = 480
bitmap = np.zeros([height, width, 3])

def accum(splits, size):
    return reduce(lambda a, b: a + [a[-1] + size * b], splits, [0])[1:]

cbar_row_heights = accum([r[0] for r in rows], height)
def cbar_row(row):
    col_widths = accum([c[0] for c in row], width)
    col_per_px = [bisect.bisect_right(col_widths, i) for i in xrange(width)]
    yiq_per_px = [colors[row[col][1]] for col in col_per_px]

    def compose(yiq, phase_offset, i):
        y, ph, sat = yiq
        y = (y - 7.5) / (100. - 7.5)
        ph = math.radians(ph - 123)
        sat = sat / 100.

        t = i / (samp_rate*1e6) #(float(i) / width) * 52.6e-6
        
        return y + sat * math.sin((t * cburst_freq + phase_offset) * 2*math.pi + ph)
    return [np.array([compose(yiq, ph_o, i) for i, yiq in enumerate(yiq_per_px)]) for ph_o in (0, .5)]
cbar_rows = [cbar_row(r[1]) for r in rows]

fps = 20
_ovltxt = None
_ovl = None
def mkoverlay(fnum):
    global _ovltxt
    global _ovl
    
    txt = ' 1996-08-29\nThu 03:51:%02d' % (6 + fnum/float(fps))
    if txt != _ovltxt:
        overlay = Image.new('LA', (width, height))
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 24)
        tw = max(font.getsize(s)[0] for s in txt.split('\n'))
        x = (width-tw)/2
        y = rows[0][0]*height - 52
        for xo in (1, -1):
            for yo in (1, -1):
                draw.text((x+xo, y+yo), txt, (0,), font=font)
        draw.text((x,y), txt, (255,), font=font)
        #overlay.save('/tmp/test.png')
        ovl = np.array(overlay)
        ovl = np.swapaxes(ovl, 1, 2)
        _ovltxt = txt
        _ovl = ovl
    return _ovl
    
passfuncs = {
    'chroma': lambda f: min(max( .5 + 2e-6*(f - (cburst_freq - chroma_bw))   ,0),1),
    'luma': lambda f: min(max( .5 + -2e-6*(f - (cburst_freq - chroma_bw))   ,0),1),
    'qam': lambda f: 1 if f < chroma_bw else 0,
}
passkernels = {}

import numpy
def bandpass(buf, passtype):
    ifft = np.fft.rfft(buf)

    key = (len(ifft), passtype)
    if key not in passkernels:
        passkernels[key] = np.array([passfuncs[passtype](float(i)/len(ifft)*(1e6*samp_rate/2)) for i in xrange(len(ifft))])
    kernel = passkernels[key]

    return np.fft.irfft(ifft * kernel)

num_frames = fps*5
for fnum in xrange(num_frames):
    for ln in xrange(height):
        phase_mode = (fnum + ln)%2
        ref_phase = phase_mode * math.pi

        #buf = [.8 if (i/100)%2 == 0 else .2 for i in xrange(667)]
        #buf = [.5 + .3*math.sin(  i/(samp_rate*1e6)*cburst_freq*2*math.pi  ) for i in xrange(667)]
        #buf = [(.8 if (i/100)%2 == 0 else .2) + .3*math.sin(  (i/(samp_rate*1e6)*cburst_freq + (0 if (i/100)%2==0 else .5) )*2*math.pi + ref_phase ) for i in xrange(width)]
        buf = np.array(cbar_rows[bisect.bisect_right(cbar_row_heights, ln)][phase_mode])

        ovl = mkoverlay(fnum)
        alpha = ovl[ln][1] / 255.
        ovlum = ovl[ln][0] / 255.
        buf = (1.-alpha)*buf + alpha*ovlum
        
        #for i in xrange(len(buf)):
        #    alpha = ovl[ln][i][1] / 255.
        #    lum = ovl[ln][i][0] / 255.
        #    buf[i] = (1.-alpha)*buf[i] + alpha*lum

        chroma_bw = 1.1e6
        chroma = bandpass(buf, 'chroma')
        luma = bandpass(buf, 'luma')
        
        isig = 2 * Iref[:len(chroma)] * chroma
        qsig = 2 * Qref[:len(chroma)] * chroma
        isig = bandpass(isig, 'qam')
        qsig = bandpass(qsig, 'qam')
        mag = (isig**2 + qsig**2)**.5
        phase = np.arctan2(-qsig, isig)

        #plt.plot(mag)
        #plt.plot(phase)
        #plt.show()

        sync_level = -47.5/92.5
        blank_level = -7.5/92.5
        ire = .01

        ix = ((np.arange(width) + .5) / width * len(luma)).astype(int)
        #print 'i', time.time() - start
        ay = luma[ix]
        ay = (ay - 1) * 92.5/(92.5+4) + 1
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
        bitmap[ln] = rgb
        #print 'm', time.time() - start

    img = Image.fromarray(bitmap.astype('uint8'), 'RGB')
    img.save('/tmp/artif%05d.png' % fnum)
    print 'wrote frame', fnum
    #print 'h', time.time() - start
        

        
