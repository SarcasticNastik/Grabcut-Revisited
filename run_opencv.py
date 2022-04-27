import numpy as np
import sys
from window_manager import WindowManager
from cv2 import grabCut

def grabcut_fn(img, mask, rect, n_itrs, mode):
    tmp1 = np.zeros((1, 65), np.float64)
    tmp2 = tmp1.copy()
    grabCut(img, mask, rect, tmp1, tmp2, n_itrs, mode)


if __name__ == '__main__':
    file = sys.argv[1] if len(sys.argv) > 1 else "messi5.jpg"
    wm = WindowManager(file, grabcut_fn)
    wm.run()

