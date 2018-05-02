import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def normalize2dpts(in_pts):
    centroid = np.array([in_pts[:, 0].mean(), in_pts[:, 1].mean(), 0])
    pts = in_pts - centroid

    meandist = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2).mean()
    scale = np.sqrt(2) / (meandist)

    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

    newpts = np.dot(T, in_pts.transpose()).transpose()
    #     result = np.dot(np.linalg.inv(T), newpts.transpose()).transpose()
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

def findFundMatRansac(matches_kp1, matches_kp2, s=8, threshold=1,
           maxIterations=5000, returnMatches=False,
           inlierThreshold=50, confidence=0.99999):
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

        # Current Inliers
        inlier_indices = [errs[:, 0] < threshold]

        # Get Current Inliers
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
                k = iter + nk

        if iter > k:
            break

#     print(str(best_total) + "/" + str(cnt_matches))
    if returnMatches:
        return best_fit, best_kp1, best_kp2

    return best_fit


def decomposeEssentialMatrix(E):
    u, s, v = np.linalg.svd(E)
    m = (s[0] + s[1]) / 2
    E = np.dot(np.dot(u, np.diag([m, m, 0])), v)

    u, s, v = np.linalg.svd(E)
    w = np.array([[0, -1, -0], [1, 0, 0], [0, 0, 1]])

    if np.linalg.det(v) < 0:
        v *= -1
    if np.linalg.det(u) < 0:
        u *= -1

    u3 = u[:, -1]
    R1 = np.dot(u, np.dot(w, v))
    R2 = np.dot(u, np.dot(w.transpose(), v))

    return [np.vstack((R1.transpose(), u3)).transpose(),
            np.vstack((R1.transpose(), -u3)).transpose(),
            np.vstack((R2.transpose(), u3)).transpose(),
            np.vstack((R2.transpose(), -u3)).transpose()]


def triangulatePoint(point_1, point_2, p1, p2):
    # define A
    u1, v1 = point_1[0], point_1[1]
    u2, v2 = point_2[0], point_2[1]

    A = np.zeros((4, 4))

    A[0] = u1 * p1[2] - p1[0]
    A[1] = v1 * p1[2] - p1[1]
    A[2] = u2 * p2[2] - p2[0]
    A[3] = v2 * p2[2] - p2[1]

    u, s, v = np.linalg.svd(A)
    x = v[-1]
    x = x / x[-1]
    return x


def triangulate(pts1, pts2, p1, p2):
    R1, t1 = getRmatAndTmat(p1)
    R2, t2 = getRmatAndTmat(p2)

    # compute camera centers
    C1 = -np.dot(R1.transpose(), t1)
    C2 = -np.dot(R2.transpose(), t2)

    V1 = np.dot(R1.transpose(), np.array([0, 0, 1]))
    V2 = np.dot(R2.transpose(), np.array([0, 0, 1]))

    points = []
    for pt1, pt2 in zip(pts1, pts2):
        point_in_3d = triangulatePoint(pt1, pt2, p1, p2)[:3]
        test1 = np.dot((point_in_3d - C1), V1)
        test2 = np.dot((point_in_3d - C2), V2)
        if (test1 > 0 and test2 > 0):
            points.append(point_in_3d)

    return np.array(points)


def getRmatAndTmat(p):
    R = p[:, :3]; t = p[:, 3];
    return R, t


def findPandX(kp1, kp2, K1, K2, p2_solutions):
    p1 = np.vstack((np.eye(3), np.zeros(3))).transpose()
    p1 = np.dot(K1, p1)
    best_p2 = -1
    best_p2_inliers = -1
    best_p2_points = []
    for sol_p2 in p2_solutions:
        p2 = np.dot(K2, sol_p2)
        points = triangulate(kp1, kp2, p1, p2)
        if (best_p2_inliers < points.shape[0]):
            best_p2_inliers = points.shape[0]
            best_p2 = p2
            best_p2_points = points
    return p1, best_p2, best_p2_points

def CheckVisible(M, P1, P2, P3):
    '''
    used to check if the surface normal facing the camera
    
    M: 3x4 projection matrix
    P1, P2, P3: 3D points
    '''
    tri_normal = np.cross((P2-P1), (P3-P2))
    # camera direction
    cam_dir = np.asarray([M[2, 0], M[2, 1], M[2, 2]]) 
    
    test_result = np.dot(cam_dir, tri_normal)
    
    if (test_result<0):
        bVisible = 1  # visible
    else:
        bVisible = 0;  # invisible
        
    return bVisible

def obj_main(P, p_img2, M, tex_name, im_index):
    tuples_img2_pts = [(p_img2[i, 0], p_img2[i, 1]) for i in range(len(p_img2))]
    img = plt.imread(tex_name)
    img_size = img.shape
    '''
    % mesh-triangulation
    '''
    tri = mtri.Triangulation(p_img2[:, 0], p_img2[:, 1])
    # trisurf mesh triangulation
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_trisurf(P[:, 0], P[:, 1], P[:, 2], triangles=tri.triangles)
    
    with open('model'+str(im_index)+'.obj', 'w') as fp:
        fp.write("# objfile\n")
        fp.write('mtllib model'+str(im_index)+'.mtl\n\n')
        fp.write('usemtl Texture\n')
        
        for i in range(len(P)):
            fp.write('v %f %f %f \n' % (P[i, 0], P[i, 1], P[i, 2]))
            
        fp.write('\n\n\n')
        
        for i in range(len(p_img2)):
            fp.write('vt %f %f\n' % (p_img2[i, 0]/img_size[1], 1-p_img2[i, 1]/img_size[0]))
            
        fp.write('\n\n\n')
        
        for i, triangle in enumerate(tri.triangles):
            bVisible = CheckVisible(M, P[triangle[0], :], P[triangle[1], :], P[triangle[2], :])
            if bVisible == True:
                fp.write('f %d/%d %d/%d %d/%d\n' % (triangle[0]+1, triangle[0]+1, triangle[1]+1, triangle[1]+1, triangle[2]+1, triangle[2]+1))
            else:
                fp.write('f %d/%d %d/%d %d/%d\n' % (triangle[1]+1, triangle[1]+1, triangle[0]+1, triangle[0]+1, triangle[2]+1, triangle[2]+1))
    
    with open('model'+str(im_index)+'.mtl', 'w') as fp:
        fp.write('# MTL file\n')
        fp.write('newmtl Texture\n')
        fp.write('Ka 1 1 1\nKd 1 1 1\nKs 1 1 1\n')
        fp.write('map_Kd '+tex_name+'\n')
