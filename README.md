# Grabcut-Revisited

Codebase and Report while reproducing the paper [GrabCut](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf).

## How to run

- Install dependencies: python packages are `opencv-python` and `igraph`.  
Conda env file has been provided to be used as `conda env create -f env.yml`.

- Run: `python3 run.py [image.jpg]`

### Controls

- Draw an rectanlge inside the input image using right click. Make sure to not to leave any part of foreground outside it.  
- Press `n` to initiate segmentation. Press repeatedly for more iteration.
- After initial segmentation, user can touch up using brushes click by left-clicks. Press the following keys to change the brush type:
  - `0`: Sure background.
  - `1`: Sure foreground.
  - `2`: Probably background.
  - `3`: Probably foreground.

- After brush strokes press `n` again to run algorithm.

- Press `r` to reset the image.

- Press `s` to save current image as `out.png`.

## Explanations and Report

Find the explanations, report and experiments done in the given [pdf](GrabCut_Algorithm.pdf).

### Project by

- Aman Rojjha (2019111018)
- Bhaskar Joshi (2019111002)
- Vedansh Mittal (2019101054)
- Utkarsh Upadhyay (2019101010)
