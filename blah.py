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
for i, s in enumerate(syncs()):
    if last_h:
        elapsed = s - last
        

        
        k = elapsed/63.55555
        epsilon = .005
        if not ((.5 - epsilon < k < .5 + epsilon) or (1 - epsilon < k < 1 + epsilon)):
            print i, elapsed
        
        #print i, elapsed
    last_h = s

"""
x = []
y = []
for i, s in enumerate(samples()):
    if i == 10000000:
        break
    x.append(i/10.5)
    y.append(s)   
 
fig, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()
"""
