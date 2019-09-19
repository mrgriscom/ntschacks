import csv
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import glob
import re

"""
Overlay the decoded video frames from ntschack.py with annotated text from the closed captioning stream

This script performs no decoding; it just composits the outputs of ntschack.py
"""

TXTSZ = 28
txtfont = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", TXTSZ)
hexfont = ImageFont.truetype("/home/drew/Downloads/UnicodeBMPFallback.ttf", TXTSZ)
txtw, txth = txtfont.getsize('A')
hexheight = hexfont.getsize('A')[1]
hexfont = ImageFont.truetype("/home/drew/Downloads/UnicodeBMPFallback.ttf", int(float(TXTSZ) * txth/hexheight))

X0 = 10
Y0 = 22
LINESEP = 1.05
BOTTOM_MARGIN = int(.75*txth)    

FRAMES = '/home/drew/nobk/ntsc/frame*.png'
CCDATA = '/home/drew/nobk/ntsc/cc.out'
OUT =    '/home/drew/nobk/ntsc/annot/framecc%05d.png'
CCLINES = 5

def frame_num(path):
    regex = FRAMES.replace('*', '([0-9]+)')
    return int(re.match(regex, path).group(1))

with open(CCDATA) as f:
    r = csv.DictReader(f)
    ccdata = dict((int(row['frame']), row['data'].decode('utf8')) for row in r if int(row['channel']) == 1)

cctext = ''
    
for path in sorted(glob.glob(FRAMES)):
    n = frame_num(path)

    ccnew = ccdata.get(n)
    if ccnew:    
        if cctext and ord(cctext[-1]) > 127 and ord(ccnew[0]) <= 127:
            cctext += '\n'
        cctext += ccnew
        cctext = '\n'.join(cctext.split('\n')[-CCLINES:])

    img = Image.open(path)
    ccheight = int(Y0 + BOTTOM_MARGIN + txth * (LINESEP * (CCLINES - 1) + 1))
    out = Image.new('RGB', (img.width, img.height+ccheight))
    out.paste(img, (0, 0, img.width, img.height))

    draw = ImageDraw.Draw(out)

    x = X0
    y = Y0 + img.height
    for i, c in enumerate(cctext):
        if ord(c) > 127:
            font = hexfont
            yo = .125*txth
            color = tuple([171]*3)
        else:
            font = txtfont
            yo = 0
            color = tuple([255]*3)

        prevc = cctext[i-1] if i > 0 else None
        if ord(c) > 127 and prevc and ord(prevc) <= 127 and prevc != '\n':
            x += .5*txtw
        if c == '\n':
            y += txth * LINESEP
            x = X0
            continue

        draw.text((x, y + yo), c, color, font=font)
        x += font.getsize(c)[0]
        
    out.save(OUT % n)
    print n


