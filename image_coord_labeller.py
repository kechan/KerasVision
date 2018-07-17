import numpy as np

import argparse, logging, os, shutil, glob

import Tkinter
from PIL import Image, ImageTk
from sys import argv

description = """Tool to help with labelling bounding box\n
E.g. \n
python image_coord_labeller.py --root_dir data/64x64_cropped_hdf5\n
"""

parser = argparse.ArgumentParser(description)

parser.add_argument('--root_dir', default='', help='Root data directory')

def left_click_callback(event):
    print "left clicked at: ", event.x, event.y

def right_click_callback(event):
    print "right clicked at: ", event.x, event.y


if __name__ == '__main__':

    args = parser.parse_args()
    root_dir = args.root_dir

    assert os.path.exists(root_dir), "Root directory does not exist." 

    window = Tkinter.Tk(className="bla")

    #image = Image.open(argv[1] if len(argv) >=2 else "bla2.png")

    image = Image.open('/Users/kelvinchan/Documents/CoinSee/data/original_and_cropped_merged_heads/validation/25c/FAR_TAIL_4vFsZZ.jpg')

    r = image.size[0]/512.0
    print(r)

    image = image.resize((512, 512), Image.ANTIALIAS)

    canvas = Tkinter.Canvas(window, width=image.size[0], height=image.size[1])
    canvas.pack()
    image_tk = ImageTk.PhotoImage(image)
    canvas.create_image(image.size[0]//2, image.size[1]//2, image=image_tk)


    canvas.bind("<Button-1>", left_click_callback)
    canvas.bind("<Button-2>", right_click_callback)

    Tkinter.mainloop()
