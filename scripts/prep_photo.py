"""
Prepare a photo for clean ASCII conversion:
  1. remove the background (rembg) so the subject is isolated
  2. boost LOCAL contrast (CLAHE) so flat lighting gains highlights and
     shadows -- this is what turns a dark blob into recognizable detail
  3. composite the subject onto pure white so the background reads as blank
     (white -> spaces in the ascii ramp)

Pass --keep-bg to skip step 3 (the white composite) and keep the whole frame
instead -- use this when the background is part of the picture (e.g. an
environmental shot) rather than something to isolate a subject from. The
subject mask from rembg is still used, just to darken/thicken the subject
relative to the background instead of erasing the background -- otherwise
subject and background end up at the same ink density and the subject
doesn't read as being in front of anything.

Output: source-prepped.png (grayscale), consumed by make_ascii_svg.py.
Run once whenever the source photo changes; the ascii SVG itself is static.

    python scripts/prep_photo.py <input.jpg> [output.png] [--keep-bg]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image
from rembg import remove

HERE = os.path.dirname(os.path.abspath(__file__))
# accept both --keep-bg and -keep-bg; ignore any other leading dash args when
# parsing positionals so flags don't become the output filename
FLAGS = {"--keep-bg", "-keep-bg"}
args = [a for a in sys.argv[1:] if a not in FLAGS and not a.startswith("-")]
KEEP_BG = any(f in sys.argv[1:] for f in FLAGS)
INP = args[0] if len(args) > 0 else os.path.join(HERE, "..", "source-photo.jpg")
OUT = args[1] if len(args) > 1 else os.path.join(HERE, "..", "source-prepped.png")

# how much darker the subject is pushed relative to the background, and how
# much lighter the background is pushed relative to the subject -- this is
# what makes the subject read as a solid foreground figure once ascii-fied
SUBJECT_DARKEN = 0.5    # multiplicative -- lower = subject inks in denser
SUBJECT_OFFSET = -10
BG_LIFT = 1.08           # multiplicative -- higher = background reads lighter
BG_OFFSET = 12
MASK_FEATHER = 2.0       # gaussian blur sigma on the subject mask edge

if KEEP_BG:
    rgba = Image.open(INP).convert("RGBA")
    orig_rgb = np.array(rgba.convert("RGB"))
    alpha = np.array(remove(rgba).split()[-1])     # 0 = background, subject mask
    mask = cv2.GaussianBlur(alpha.astype(np.float32) / 255.0, (0, 0), MASK_FEATHER)

    gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.05, beta=18).astype(np.float32)

    subject = np.clip(gray * SUBJECT_DARKEN + SUBJECT_OFFSET, 0, 255)
    background = np.clip(gray * BG_LIFT + BG_OFFSET, 0, 255)
    out = subject * mask + background * (1.0 - mask)
    out = np.clip(out, 0, 255).astype(np.uint8)
else:
    # 1. cut out the subject
    cut = remove(Image.open(INP).convert("RGBA"))
    rgb = np.array(cut.convert("RGB"))
    alpha = np.array(cut.split()[-1])             # 0 = background

    # 2. local-contrast the luminance (CLAHE)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # a touch of global lift so the subject sits in the sparse end of the ramp
    gray = cv2.convertScaleAbs(gray, alpha=1.05, beta=18)

    # 3. paste onto white using the alpha mask (feathered a hair to avoid a halo)
    mask = (alpha.astype(np.float32) / 255.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
    out = gray.astype(np.float32) * mask + 255.0 * (1.0 - mask)
    out = np.clip(out, 0, 255).astype(np.uint8)

Image.fromarray(out, mode="L").save(OUT)
print("wrote", OUT, out.shape)
