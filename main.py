from lib.util import *
from lib.hw3 import *
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description="CV HW2 image stitching")
parser.add_argument('-img1', type=str, default='./data/Mesona1.JPG')
parser.add_argument('-img2', type=str, default='./data/Mesona2.JPG')

args = parser.parse_args()

# Reading Data and Get the Gray values only
img1 = cv.imread(args.img1,0)
img2 = cv.imread(args.img2,0)


# Make SIFT Object
sift = cv.xfeatures2d.SIFT_create()

# Detect Local Descriptors and Keypoints
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# Brute Force Local Descripter Matcher
bf = cv.BFMatcher(cv.NORM_L2)

# For each local descriptor des1, find 2 best correspondence from des 2
matches = bf.knnMatch(des1, des2, k=2)  # Think of it just finding a cluster of ssd.

# Filter those correspondences using ratio test
matches, kp1, kp2 = filter_ratio_matches(matches, kp1, kp2, 0.75)

# Show current correspondences
output_2 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
show_image(cv.cvtColor(output_2, cv.COLOR_BGR2RGB))


# findFundamentalMatrix using RANSAC
matches_kp1, matches_kp2 = transformPtsToArrayPts(kp1, kp2, matches)
best_fit, best_kp1, best_kp2 = findFundMatRansac(matches_kp1, matches_kp2, returnMatches=True)

# The first result
lines1 = computeEpipoleLines(best_fit.transpose(), best_kp2)
img3,img4 = drawlines(img1,img2,lines1,best_kp1,best_kp2)
show_comparison_image(img3, img4)

# The first result
lines2 = computeEpipoleLines(best_fit, best_kp1)
img5,img6 = drawlines(img2,img1,lines2,best_kp2,best_kp1)
show_comparison_image(img5, img6)

