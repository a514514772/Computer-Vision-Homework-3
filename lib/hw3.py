import cv2 as cv
import numpy as np
import random


def normalize2dpts(in_pts):
    centroid = np.array([in_pts[:, 0].mean(), in_pts[:, 1].mean(), 0])
    pts = in_pts - centroid

    meandist = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2).mean()
    scale = np.sqrt(2) / (meandist)

    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

    newpts = np.dot(T, in_pts.transpose()).transpose()
    return newpts, T


def fundmat(in_points_in_img1, in_points_in_img2):
    # normalize
    points_in_img1, T1 = normalize2dpts(in_points_in_img1)
    points_in_img2, T2 = normalize2dpts(in_points_in_img2)

    # Solve for A
    s = points_in_img1.shape[0]

    A = np.zeros((s, 9))
    for index in range(0, s):
        x, y = points_in_img1[index][0], points_in_img1[index][1]
        tx, ty = points_in_img2[index][0], points_in_img2[index][1]
        A[index] = [tx * x, tx * y, tx, ty * x, ty * y, ty, x, y, 1]

    u, s, v = np.linalg.svd(A)
    F = v[-1].reshape(3, 3)  # eigenvector with the least eigenvalue

    u, s, v = np.linalg.svd(F)
    s[2] = 0
    F = np.dot(np.dot(u, np.diag(s)), v)

    # denormalize
    F = np.dot(np.dot(T2.transpose(), F), T1)

    return F / F[2, 2]


def computeEpipoleLines(F, pts):
    lines = np.dot(F, pts.transpose()).transpose()
    n = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2).reshape(-1, 1)
    return lines / n * -1


def transformPtsToArrayPts(kp1, kp2, matches):
    tup_matches_kp1 = [kp1[dt[0].queryIdx].pt for dt in matches]
    tup_matches_kp2 = [kp2[dt[0].trainIdx].pt for dt in matches]
    matches_kp1 = np.array([[h for h in kp] + [1] for kp in tup_matches_kp1])
    matches_kp2 = np.array([[h for h in kp] + [1] for kp in tup_matches_kp2])
    return matches_kp1, matches_kp2


def calculateSampsonDistance(matches_kp1, matches_kp2, F):
    Fx1 = np.dot(F, matches_kp1.transpose())
    Fx2 = np.dot(F.transpose(), matches_kp2.transpose())
    denom = (Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2).reshape(-1, 1)

    #     print(denom.shape)
    err = (np.diag(np.dot(matches_kp2, np.dot(F, matches_kp1.transpose()))) ** 2)
    err = err.reshape(-1, 1) / denom
    return err


def randomPartition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


def findFundMatRansac(matches_kp1, matches_kp2, s=8, threshold=3,
           maxIterations=5000, returnMatches=False,
           inlierThreshold=50, confidence=0.99):
    cnt_matches = matches_kp1.shape[0]
    best_fit = []
    best_error = np.Infinity
    best_kp1, best_kp2 = [], []
    best_total = 0

    k = maxIterations
    for iter in range(k):
        maybe_idxs, test_idxs = randomPartition(s, cnt_matches)
        # Take s data points
        data_p1 = np.take(matches_kp1, maybe_idxs, axis=0)
        data_p2 = np.take(matches_kp2, maybe_idxs, axis=0)
        # Fit a fundamental matrix
        F = fundmat(data_p1, data_p2)

        # Test the current fundamental matrix
        test_p1 = np.take(matches_kp1, test_idxs, axis=0)
        test_p2 = np.take(matches_kp2, test_idxs, axis=0)
        errs = calculateSampsonDistance(test_p1, test_p2, F)

        # Initialize Current Matches
        current_p1, current_p2 = [], []
        current_total = 0

        inlier_indices = [errs[:, 0] < threshold];
        current_p1 = np.append(data_p1, test_p1[inlier_indices], axis=0)
        current_p2 = np.append(data_p2, test_p2[inlier_indices], axis=0)
        current_total = current_p1.shape[0]

        if current_total > best_total and current_total >= inlierThreshold:
            better_fit = fundmat(current_p1, current_p2)
            better_err = calculateSampsonDistance(current_p1, current_p2, F)

            if (best_error > better_err.mean()):
                best_fit = better_fit
                best_kp1 = current_p1
                best_kp2 = current_p2
                best_total = current_p1.shape[0]

                # # we are done in case we have enough inliers
                r = current_total / cnt_matches
                nk = np.log(1 - confidence) / np.log(1 - pow(r, s))
                if iter > nk:
                    break

    print(str(best_total) + "/" + str(cnt_matches))
    if returnMatches:
        return best_fit, best_kp1, best_kp2

    return best_fit
