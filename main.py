from lib.util import *
from lib.hw3 import *
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import argparse

parser = argparse.ArgumentParser(description="CV HW3 SfM")
parser.add_argument('-mode', type=int, default=2)
args = parser.parse_args()

# Reading Data and Get the Gray values only
img1 = cv.imread("data/Mesona1.JPG",0)
img2 = cv.imread("data/Mesona2.JPG",0)
K1 = np.array([[1.4219, 0.0005, 0.5092],[0, 1.4219, 0.3802],[0,0,0.0010]])
K2 = np.array([[1.4219, 0.0005, 0.5092],[0, 1.4219, 0.3802],[0,0,0.0010]])

if args.mode == 2:
    img1 = cv.imread("data/Statue1.bmp", 0)
    img2 = cv.imread("data/Statue2.bmp", 0)
    K1 = np.array([[5426.566895, 0.678017, 330.096680], [0.000000, 5423.133301, 648.950012], [0, 0, 1]])
    K2 = np.array([[5426.566895, 0.678017, 387.430023], [0.000000, 5423.133301, 620.616699], [0, 0, 1]])

show_img1 = cv.cvtColor(img1, cv.COLOR_GRAY2RGB)
show_img2 = cv.cvtColor(img2, cv.COLOR_GRAY2RGB)
show_comparison_image(show_img1,show_img2, "Image1", "Image2")

"""
    
    Find The Correspondences
    
"""
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
bf = cv.BFMatcher(cv.NORM_L2)
# For each local descriptor des1, find 2 best correspondence from des 2
matches = bf.knnMatch(des1, des2, k=2)
# Filter those correspondences using ratio test
matches, kp1, kp2 = filter_ratio_matches(matches, kp1, kp2, 0.7)

output = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
show_image(output)
"""
    
    Find The Fundamental Matrix
    
"""
matches_kp1, matches_kp2 = transformPtsToArrayPts(kp1, kp2, matches)
F, best_kp1, best_kp2 = findFundMatRansac(matches_kp1, matches_kp2, returnMatches=True, threshold=0.1)

# Show Our Correspondences
lines1 = computeEpipoleLines(F.transpose(), best_kp2)
img3,img4 = drawlines(img1,img2,lines1,best_kp1,best_kp2)
lines2 = computeEpipoleLines(F, best_kp1)
img6,img5 = drawlines(img2,img1,lines2,best_kp2,best_kp1)
'''
plt.figure()
plt.tight_layout()
plt.subplot(221); plt.imshow(img3); plt.xticks([]), plt.yticks([])
plt.subplot(222); plt.imshow(img4); plt.xticks([]), plt.yticks([])
plt.subplot(223); plt.imshow(img5); plt.xticks([]), plt.yticks([])
plt.subplot(224); plt.imshow(img6); plt.xticks([]), plt.yticks([])
plt.show()
'''
"""
    Computing Essential Matrix
    
"""

E = np.dot(K2.transpose(), np.dot(F, K1))


"""
    Find The Best Camera Matrix P and 3d-Points
"""
all_p2_solutions = decomposeEssentialMatrix(E)
best_p1, best_p2, points = findPandX(best_kp1, best_kp2, K1, K2, all_p2_solutions)

if args.mode == 2:
	obj_main(points, best_kp1, best_p1, "data/Statue1.bmp", 2)
else:
	obj_main(points, best_kp1, best_p1, "data/Mesona1.JPG", 1)

plt.rcParams['figure.figsize'] = [8,8]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')
plt.show()