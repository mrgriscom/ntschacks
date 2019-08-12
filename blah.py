import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def samples():
    with open('cvbs.i16') as f:
        while True:
            buf = f.read(2)
            if not buf:
                return
            yield -struct.unpack('<H', buf)[0]

# TODO track sync len            
def syncs():
    lockout = 200000
    debounce = 10
    sync_thresh = -800
    last_sync = 1e100 # set to infinity so no sync registered until we see an above-sync sample first
    for i, val in enumerate(samples()):
        if i < lockout:
            continue

        if val < sync_thresh:
            if last_sync is None:
                last_sync = i
            elif i - last_sync == debounce:
                yield last_sync/10.5
        else:
            last_sync = None

def within(pct, epsilon=.005):
    return 1. - epsilon < pct < 1. + epsilon
            
last_h = None
hsyncs = []
for i, s in enumerate(syncs()):
    #if s > 4e6: # just process beginning to start
    #    break

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

front_porch = 1.5
back_porch = 9.4  # includes sync pulse
padding = .005*63.555
#padding = .015*63.555

sync_level = -920
blank_level = -690
ire = (blank_level - sync_level) / 40.
white_level = blank_level + 100*ire

sync_ix = 0
def calc_sync():
    sync_us = hsyncs[sync_ix]
    return (sync_us - front_porch - padding) * 10.5, (sync_us + back_porch + padding) * 10.5
sync_start, sync_end = calc_sync()
x = []
y = []

out1 = open('cvbs-scram.i16', 'w')
out2 = open('cvbs-scram-inv.i16', 'w')

#duty_cycle = [2, 2]
duty_cycle = [0, 2]
def scramble(t):
    t *= 1e-6
    return t % (duty_cycle[0] + duty_cycle[1]) > duty_cycle[0]

for i, val in enumerate(samples()):
    val1 = val
    val2 = val
    val = None
    
    t = i/10.5
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

"""
fig, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()
"""
