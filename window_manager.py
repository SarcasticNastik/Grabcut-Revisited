import cv2
import numpy as np
import types
import cv2 as cv
import constants
from constants import Colors, MaskValues
# from constants import Colors, MaskValues

BRUSH_MODES = {
    "bg": {"col": Colors.BLACK, "mask_val": MaskValues.bg},
    "fg": {"col": Colors.WHITE, "mask_val": MaskValues.fg},
    "pr_bg": {"col": Colors.RED, "mask_val": MaskValues.pr_bg},
    "pr_fg": {"col": Colors.GREEN, "mask_val": MaskValues.pr_fg},
}

RECT_COLOR = Colors.BLUE


class WindowManager:
    INPUT_WINDOW = 'input'
    OUTPUT_WINDOW = 'output'

    def __init__(self, img_path: str, grabcut_fn):
        self.grabcut_fn = grabcut_fn
        self.org_img = cv.imread(img_path)
        self.disp_img = self.org_img.copy()
        self.out_img = np.zeros(self.org_img.shape, np.uint8)
        self.mask = np.zeros(self.org_img.shape[:2], np.uint8)
        cv.namedWindow(self.INPUT_WINDOW, cv.WINDOW_AUTOSIZE |
                       cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)
        cv.namedWindow(self.OUTPUT_WINDOW, cv.WINDOW_AUTOSIZE |
                       cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)
        cv.setMouseCallback(self.INPUT_WINDOW, self.mouse_callback)
        cv.moveWindow(self.INPUT_WINDOW, self.org_img.shape[1]+110, 0)
        self.reset_state()

    def run(self):
        key_to_mode = {
            '0': BRUSH_MODES["bg"],
            "1": BRUSH_MODES["fg"],
            "2": BRUSH_MODES["pr_bg"],
            "3": BRUSH_MODES["pr_fg"],
        }

        key_to_act = {
            "s": self.save_output,
            "r": self.reset_state,
            "n": self.perform_segmentation,
            "b": self.toggle_rect_draw_mode,
        }

        while True:
            hard_mask = (self.mask == 1) | (self.mask == 3)
            self.out_img = self.org_img*hard_mask[:, :, None]

            cv.imshow(self.OUTPUT_WINDOW, self.out_img)
            cv.imshow(self.INPUT_WINDOW, self.disp_img)
            k = cv.waitKey(1)
            if k < 0:
                continue

            k = chr(k)

            if k in key_to_mode:
                self.draw_ns.brush_mode = key_to_mode[k]
            if k in key_to_act:
                key_to_act[k]()
            elif k == "q":
                cv.destroyAllWindows()
                return

    def perform_segmentation(self):
        #updates the mask inplace to corresponding segmentation
        self.grabcut_fn(self.org_img,
                        self.mask,
                        self.rect,
                        constants.N_ITERS,
                        self.init_mode
                        )
        self.init_mode = cv.GC_INIT_WITH_MASK

    def toggle_rect_draw_mode(self):
        self.draw_ns.rect_on = not self.draw_ns.rect_on

    def save_output(self, file_path: str = "out.png"):
        cv.imwrite(file_path, self.out_img)
        print("Output image saved at ", file_path)

    def reset_state(self):
        self.disp_img = self.org_img.copy()
        self.rect = (0, 0, 1, 1)
        self.init_mode = cv.GC_INIT_WITH_RECT
        self.drawing = True
        self.rectangle = False
        self.draw_ns = types.SimpleNamespace()
        self.draw_ns.rect_on = False
        self.draw_ns.brush_on = False
        self.draw_ns.ix = 0
        self.draw_ns.iy = 0
        self.draw_ns.brush_sz = 3
        self.draw_ns.brush_mode = BRUSH_MODES["bg"]

    def mouse_callback(self, event, x, y, flags, param):
        def mark_circle():
            cv.circle(self.disp_img, (x, y), self.draw_ns.brush_sz,
                      self.draw_ns.brush_mode['col'], -1)
            cv.circle(self.mask, (x, y), self.draw_ns.brush_sz,
                      self.draw_ns.brush_mode['mask_val'], -1)

        def mark_rectangle(set_rect=False):
            ix, iy = self.draw_ns.ix, self.draw_ns.iy
            img = self.org_img.copy()
            cv.rectangle(img, (ix, iy), (x, y), RECT_COLOR, 2)
            self.disp_img = img

            if set_rect:
                # left, up, width, height
                self.rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))

        if event == cv.EVENT_RBUTTONDOWN:
            self.draw_ns.rect_on = True
            self.draw_ns.ix, self.draw_ns.iy = x, y
            self.init_mode = cv.GC_INIT_WITH_RECT

        elif event == cv.EVENT_RBUTTONUP:
            if self.draw_ns.rect_on:
                self.draw_ns.rect_on = False
                mark_rectangle(set_rect=True)
                self.init_mode = cv.GC_INIT_WITH_RECT

        elif event == cv.EVENT_MOUSEMOVE:
            if self.draw_ns.rect_on:
                mark_rectangle()

            elif self.draw_ns.brush_on:
                mark_circle()

        elif event == cv.EVENT_LBUTTONDOWN:
            if self.draw_ns.rect_on:
                print("Draw the Rect first")
                return
            self.draw_ns.brush_on = True
            mark_circle()

        elif event == cv.EVENT_LBUTTONUP:
            if self.draw_ns.rect_on:
                print("Draw the Rect first")
                return
            if self.draw_ns.brush_on:
                self.draw_ns.brush_on = False
                mark_circle()
                self.init_mode = cv.GC_INIT_WITH_MASK

