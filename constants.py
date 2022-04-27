from functools import wraps
from time import time

# Tweakables
GAMMA = 50
N_ITERS = 1
GMM_COMPONENTS = 5
ROW=True
COL=True
LD=True
RD=True
EPS=10**-10

class Colors:
    RED = [255,0,0]
    GREEN = [0, 255, 0]
    BLUE = [0, 0, 255]
    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]


class MaskValues:
    bg = 0
    fg = 1
    pr_bg = 2
    pr_fg = 3

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        fname = f.__name__
        print(f"||{fname} STARTED||")
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"||{fname} ENDED in {round(te-ts,2)} sec||")
        return result
    return wrap
