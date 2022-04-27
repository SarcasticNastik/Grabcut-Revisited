import sys
from window_manager import WindowManager
from grabcut import grabcut

def grabcut_fn(img, mask, rect, n_itrs, mode):
    if mode==1:
        rect=None
    mask[:] = grabcut(img,mask,rect, n_itrs)

# For notebooks
def sim(file=None):
    file = file if file is not None else "messi5.jpg"
    wm = WindowManager(file, grabcut_fn)
    wm.run()

if __name__ == '__main__':
    file = sys.argv[1] if len(sys.argv) > 1 else "messi5.jpg"
    sim(file=file)
