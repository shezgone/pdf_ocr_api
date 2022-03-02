import pdfplumber
from sklearn.cluster import OPTICS
import numpy as np
import cv2
import pandas as pd

KEEP_BLANK_CHARS = False
N_SAMPLES = 3

if __name__ == "__main__":
    with pdfplumber.open("samples/sample4.pdf") as pdf:
        p0 = pdf.pages[0]
        im = p0.to_image()
        words_obj1 = p0.extract_words(keep_blank_chars=KEEP_BLANK_CHARS,
                                     #x_tolerance=10,
                                     #y_tolerance=20
                                     )[:-1]
        words_obj2 = p0.extract_words(keep_blank_chars=KEEP_BLANK_CHARS,
                                      # x_tolerance=10,
                                      # y_tolerance=20
                                      )[1:]

        color = ["red","blue","green","black","yellow","grey","red","blue","green","black"]
        bbox = []
        for i, (item1, item2) in enumerate(zip(words_obj1, words_obj2)):
            if item1['text'] == "X" and item2['text'] == ")":
                im.draw_rects((item1['x0'], item1['top'], item2['x1'], item2['bottom']), stroke=color[i])

        #im.draw_rects(bbox)

        im.save("samples/sample_out.png")
        cim = cv2.imread("samples/sample_out.png")
        cv2.imshow("output", cim)
        cv2.waitKey(0)
