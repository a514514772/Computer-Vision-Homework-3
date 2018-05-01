import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


def show_image(img, title=None):
    plt.figure()
    plt.imshow(img)
    if title != None:
        plt.title(title)
    plt.xticks([]), plt.yticks([])

    plt.show()


def show_comparison_image(img1, img2, title1 = None, title2 = None):
    plt.figure()
    plt.subplot(121)
    plt.imshow(img1)
    if (title1 != None):
        plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(img2)
    if (title2 != None):
        plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()


def filter_ratio_matches(matches, kp1, kp2, ratio=0.7):
    new_kp1, new_kp2, new_matches = [], [], []
    ctr = 0
    for i, (m, n) in enumerate(matches):  #
        if m.distance < ratio * n.distance:
            new_kp1.append(kp1[m.queryIdx])
            new_kp2.append(kp2[m.trainIdx])
            new_matches.append([cv.DMatch(ctr, ctr, m.distance)])
            ctr += 1
    return new_matches, new_kp1, new_kp2


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    t_pts1 = np.int32(pts1)
    t_pts2 = np.int32(pts2)
    for r, pt1, pt2 in zip(lines, t_pts1, t_pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1,lineType=4)
        img1 = cv.circle(img1,tuple(pt1[0:2]),5,color,-1)
        img2 = cv.circle(img2, tuple(pt2[0:2]), 5, color,-1)
    return img1, img2