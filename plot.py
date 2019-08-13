import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#samp_rate = 10.5
samp_rate = 14.

def samples():
    with open('/tmp/cvbs.i16') as f:
        while True:
            buf = f.read(2)
            if not buf:
                return
            yield -struct.unpack('<H', buf)[0]

x = []
y = []

for i, val in enumerate(samples()):
    if i > 20e6:
        break
    
    t = i/samp_rate
    x.append(t)
    y.append(val)

fig, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()

