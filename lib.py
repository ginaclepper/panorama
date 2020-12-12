# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 10:49:58 2020

@author: ginac
"""

import numpy as np
from cv2 import cv2
import math
import random
from scipy.optimize import least_squares
import os

# Global parameters

# Number of kp returned by SIFT. Without a limit, returns ~5000 pts.
n_sift_pts = 2000

# (n) Number of points used per trial in RANSAC
n = 6

# Number of expected false matches
epsilon = 0.4
# Can try 0.1 - 0.4

# Probability a good homography is among N RANSAC trials
p = 0.99
# Can try 0.999

# Number of RANSAC trials
N = int(np.log(1-p) / np.log(1-(1-epsilon)**n))

# Inlier threshold
sigma = 1
delta = 3*sigma

# Number of inliers needed
# M = n_total*(1-epsilon)
# I used n_best=50

def load_imgs(directory):
    imgs = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory,filename))
        if img is not None:
            imgs.append(img)
    return imgs

def display_image(win_id, image, scale):
    '''displays image until key is pressed'''
    smallimage = cv2.resize(image, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
    cv2.imshow(win_id, smallimage)
    # wait for user to press any key; prevents kernel from crashing
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def use_sift(gray_img):
    '''Find interest points using SIFT'''
    sift = cv2.xfeatures2d.SIFT_create(n_sift_pts)

    kp, des = sift.detectAndCompute(gray_img, None)
    sift_img = cv2.drawKeypoints(gray_img, kp, outImage=np.array([]),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    pts = np.zeros([len(kp),2])
    for idx, keypoint in enumerate(kp):
        pts[idx, 0] = keypoint.pt[0]
        pts[idx, 1] = keypoint.pt[1]
    return pts, des, sift_img


def find_corr_ssd_or_ncc(img1, img2, corners1, corners2, des1, des2, type, n_best=50, win=21):
    '''use SDD or NCC to find correspondences among given interest points'''
    class corrs:
        def __init__(self, pts, dist):
            self.pts = pts
            self.dist = dist
    
    # Identify smaller of two lists of corners:
    if len(corners1) < len(corners2):
        small_img_id = 1
        small_img = img1
        large_img = img2
        small_corners = corners1
        large_corners = corners2
        small_des = des1
        large_des = des2
    else:
        small_img_id = 2
        small_img = img2
        large_img = img1
        small_corners = corners2
        large_corners = corners1
        small_des = des2
        large_des = des1

    # For each descriptor in smaller list, find descriptor in larger list
    # with smallest SSD or NCC
    correspondences = []
    for i, small_corner in enumerate(small_corners):
        # Initialize min SSD or max NCC
        if (type == 'ssd'):
            best_val = np.Inf
        elif (type == 'ncc'):
            best_val = -1*np.Inf
        for j, large_corner in enumerate(large_corners):
            # Calculate SSD or NCC:
            if (type == 'ssd'):
                ssd_val = ssd(small_img, large_img, small_corner, large_corner, win)
                if ssd_val <= best_val:
                    best_val = ssd_val
                    best_pt = large_corner
            elif (type == 'ncc'):
                ncc_val = ncc(small_img, large_img, small_corner, large_corner, small_des[i], large_des[j], win)
                if ncc_val > best_val:
                    best_val = ncc_val
                    best_pt = large_corner           
        # Correspondences will always go from 1st image to 2nd image
        if(small_img_id == 1): # small image is 1st image
            correspondences.append(corrs([small_corner, best_pt], best_val))
        else: # small image is 2nd image
            correspondences.append(corrs([best_pt, small_corner], best_val))
    # Sort correspondences by SSD or NCC and only return n_best ones
    correspondences = sorted(correspondences, key = lambda x:x.dist)
    if (type == 'ncc'):
        correspondences.reverse()
    corr_result = []
    corr_output = []
    for i in range(len(correspondences)):
        corr_result.append(correspondences[i].pts)
    corr_output = corr_result[:n_best]
    return corr_output


def ssd(img1, img2, pt1, pt2, win):
    '''calculate ssd'''
    kernel1, kernel2 = ssd_ncc_kernel(img1, img2, pt1, pt2, win)
    return np.sum((kernel1 - kernel2)*(kernel1 - kernel2))


def ncc(img1, img2, pt1, pt2, des1, des2, win):
    '''calculate ncc'''
    kernel1 = des1
    kernel2 = des2
    m1 = np.mean(kernel1)
    m2 = np.mean(kernel2)
    term1 = kernel1 - m1
    term2 = kernel2 - m2
    numerator = np.sum(term1*term2)
    denominator = np.sqrt(np.sum(term1*term1)*np.sum(term2*term2))
    return numerator / denominator


def ssd_ncc_kernel(img1, img2, pt1, pt2, win):
    '''returns two kernels'''
    # Find bounds of neighborhood (in form 'from : to' where to is excluded)
    halfwin = int(win/2)
    height1 = img1.shape[0]
    width1 = img1.shape[1]
    height2 = img2.shape[0]
    width2 = img2.shape[1]
    # pt[0] is x, pt[1] is y
    if (pt1[0] - halfwin) >= 0:
        pt1_left = pt1[0] - halfwin
    else:
        pt1_left = 0
    if (pt1[0] + halfwin) < width1:
        pt1_right = pt1[0] + halfwin
    else:
        pt1_right = width1
    if (pt1[1] - halfwin) >= 0:
        pt1_low = pt1[1] - halfwin
    else:
        pt1_low = 0
    if (pt1[1] + halfwin) < height1:
        pt1_high = pt1[1] + halfwin
    else:
        pt1_high = height1
    # pt2:
    if (pt2[0] - halfwin) >= 0:
        pt2_left = pt2[0] - halfwin
    else:
        pt2_left = 0
    if (pt2[0] + halfwin) < width2:
        pt2_right = pt2[0] + halfwin
    else:
        pt2_right = width2
    if (pt2[1] - halfwin) >= 0:
        pt2_low = pt2[1] - halfwin
    else:
        pt2_low = 0
    if (pt2[1] + halfwin) < height2:
        pt2_high = pt2[1] + halfwin
    else:
        pt2_high = height2
    # If neighborhoods are different sizes (one hits edge but other doesn't), shrink larger kernel
    xdiff = (pt1_right - pt1_left) - (pt2_right - pt2_left)
    if(xdiff > 0):  # kernel 1 is wider
        pt1_right = pt1_right - xdiff
    elif(xdiff < 0):  # kernel 2 is wider
        pt2_right = pt2_right + xdiff
    ydiff = (pt1_high - pt1_low) - (pt2_high - pt2_low)
    if(ydiff > 0):  # kernel 1 is taller
        pt1_high = pt1_high - ydiff
    elif(ydiff < 0):  # kernel 2 is taller
        pt2_high = pt2_high + ydiff
    
    kernel1 = img1[pt1_low : pt1_high, pt1_left : pt1_right]
    kernel2 = img2[pt2_low : pt2_high, pt2_left : pt2_right]

    return kernel1, kernel2

def RANSAC(img1, img2, corrs):
    '''Finds best homography based on num inliers by forming H for n random points N times'''
    n_total = len(corrs)
    M = n_total*(1-epsilon)
    best_num_inliers = 0
    best_H = []
    best_inlier_corrs = []
    best_outlier_corrs = []
    for trial in range(N):
        # Run single RANSAC trial to calculate H
        H = RANSAC_trial(corrs)
        num_inliers, inlier_corrs, outlier_corrs = find_liers(H, corrs)
        if(num_inliers >= best_num_inliers):
            best_num_inliers = num_inliers
            best_H = H
            best_inlier_corrs = inlier_corrs
            best_outlier_corrs = outlier_corrs
    # Check if any H has a passable number of inliers
    if(best_num_inliers < M):
        print("Desired number of inliers M not reached.")
    # Refine chosen H by using all inlier pts
    refined_H = find_H(best_inlier_corrs[:,0:1].flatten(), best_inlier_corrs[:,1:2].flatten())
    refined_num_inliers, refined_inlier_corrs, refined_outlier_corrs = find_liers(refined_H, corrs)
    return refined_num_inliers, refined_H, refined_inlier_corrs, refined_outlier_corrs


def find_liers(H, corrs):
    '''Given H and corrs, find inliers and outliers'''
    # Map correspondence orig pts using H
    orig_pts = np.asarray(corrs)[:,0:1].reshape([len(corrs),2])
    mapped_pts = map_pts(H, orig_pts)
    # Calculate distance between mapped points to true prime points
    prime_pts = np.asarray(corrs)[:,1:2].reshape([len(corrs),2])
    diff = (mapped_pts - prime_pts)**2
    d = np.sqrt(diff[:,0:1] + diff[:,1:2])
    # Threshold distances using delta
    pass_idxs = np.where(d < delta)
    # Find number inliers
    pass_idxs = pass_idxs[0]
    num_inliers = len(pass_idxs)
    # Find inlier correspondences
    inlier_orig_pts = np.asarray([orig_pts[i] for i in pass_idxs])
    inlier_mapped_pts = np.asarray([mapped_pts[i] for i in pass_idxs])
    inlier_corrs = np.stack((inlier_orig_pts, inlier_mapped_pts), axis=1)
    # Find outlier correspondences
    fail_idxs = np.where(d >= delta)
    fail_idxs = fail_idxs[0]
    outlier_orig_pts = np.asarray([orig_pts[i] for i in fail_idxs])
    outlier_mapped_pts = np.asarray([mapped_pts[i] for i in fail_idxs])
    outlier_corrs = np.stack((outlier_orig_pts, outlier_mapped_pts), axis=1)
    return num_inliers, inlier_corrs, outlier_corrs

def RANSAC_trial(corrs):
    '''Calculates a homography for a single RANSAC trial'''
    # Calculate n random indices without duplication
    all_idxs = np.asarray(range(0, len(corrs)))
    random.shuffle(all_idxs)
    idxs = all_idxs[:n]
    # Index correspondences to yield nx2x2 array
    r_corrs = np.asarray([corrs[i] for i in idxs])
    # Calculate homography
    H = find_H(r_corrs[:,0:1].flatten(), r_corrs[:,1:2].flatten())
    return H


# orig_pts: m (x,y) points
# 1x2m array: [x1, y1, x2, y2, x3, y3, x4, y4, ...]
#
# prime_pts: m (x',y') points
# 1x2m array: [x1', y1', x2', y2', x3', y3', x4', y4', ...]
def find_H(orig_pts, prime_pts):
    '''returns 3x3 matrix H, the homography'''
    # Find A and A^{-1}
    m = int(len(orig_pts)/2)
    A = np.zeros([2*m, 9], dtype=float)
    for i in range(m):
        A[2*i] = np.array([orig_pts[2*i], orig_pts[2*i+1], 1, 0, 0, 0,
                           -orig_pts[2*i]*prime_pts[2*i],
                           -orig_pts[2*i+1]*prime_pts[2*i],
                           -prime_pts[2*i]])
        A[2*i+1] = np.array([0, 0, 0, orig_pts[2*i], orig_pts[2*i+1], 1,
                             -orig_pts[2*i]*prime_pts[2*i+1],
                             -orig_pts[2*i+1]*prime_pts[2*i+1],
                             -prime_pts[2*i+1]])

    # h33 inclusive / SVD: includes possibility h33 might be 0
    AT = np.transpose(A)
    U, D, UT = np.linalg.svd(np.matmul(AT, A))
    smallest_eigenvalue = min(D)
    column = -1
    for idx, d in enumerate(D):
        if (d == smallest_eigenvalue):
            column = idx
    hvec = U[:, column]
    H = np.reshape(hvec, [3, 3])

    return H


def refine_H(H, corrs):
    '''Uses non-linear least squares to minimize cost function and improve H'''
    # Find H that yields smallest cost
    p0 = np.reshape(H, [9,1]).flatten()
    p_new = least_squares(cost, p0, args=(corrs, 1), method='lm')
    return np.reshape(p_new.x, [3, 3])

def cost(H, corrs, extra):
    '''Computes cost for given H (and N correspondences)'''
    # Minimization proceeds with respect to its first argument
    # The argument H passed to this function must be an ndarray of shape (9,)
    # [h11; h12; h13; ... ; h33]
    # The function must allocate and return a 1-D array_like of shape (N,) or a scalar
    # We will return 2N cost components for N correspondences
    # Reshape H into 3x3 for map_pts function
    H = np.reshape(H, [3, 3])
    # Map correspondence orig pts using H
    orig_pts = np.asarray(corrs)[:,0:1].reshape([len(corrs),2])  # (N,2)
    f_pts = map_pts(H, orig_pts)  # (N,2)
    # Calculate distance between mapped points to true prime points
    prime_pts = np.asarray(corrs)[:,1:2].reshape([len(corrs),2])  # (N,2)
    diff = (prime_pts - f_pts)**2  # (N,2)
    cost = np.reshape(diff, [len(diff)*2, 1]).flatten()  # (2N,)
    return cost


def find_scale(H, orig_image, hi=750, wi=750, corners=np.array([[-999],
                                                                [-999]]),
                                                                mode='single',
                                                                pan_scale=1,
                                                                canvas=None):
    '''computes min's, max's, h0, w0, and scale'''
    # Find dimensions of new image:
    orig_length = orig_image.shape[1]
    orig_height = orig_image.shape[0]

    # Initialize min/max
    if(corners[0, 0] == -999):
        corners = np.array([[0, 0], [orig_length, 0], [0, orig_height],
                            [orig_length, orig_height]])
    x = np.matmul(H, [[0], [0], [1]])
    x = np.reshape(x, [1, -1]).flatten()
    min_x = x[0]/x[2]
    max_x = min_x
    min_y = x[1]/x[2]
    max_y = min_y
    # Min/max dimensions can be calculated from corner row-col pairs:
    for corner in corners:
        x = np.matmul(H, np.array([[corner[0]], [corner[1]], [1]]))
        x = np.reshape(x, [1, -1]).flatten()
        xcoord = x[0]/x[2]
        ycoord = x[1]/x[2]
        if (xcoord < min_x):
            min_x = xcoord
        elif (xcoord > max_x):
            max_x = xcoord
        if (ycoord < min_y):
            min_y = ycoord
        elif (ycoord > max_y):
            max_y = ycoord
    h0 = max_y-min_y
    w0 = max_x-min_x

    # Calculate scale factor
    if(mode == 'single'):
        if (h0 > w0):  # output is taller than it is long
            scale = h0/hi
        else:
            scale = w0/wi
    elif(mode == 'panorama'): # use a set scale (so it matches the other images)
        scale = pan_scale
    # If canvas already created, use that instead
    if(canvas is None):
        mapped_image = np.zeros([math.ceil(h0/scale), math.ceil(w0/scale), 3])
    else:
        mapped_image = canvas.copy()
    return min_x, max_x, min_y, max_y, scale, mapped_image

def map_pts(H, pts): # Vectorized :)
    '''maps n_total select points, in form nx2'''
    n_total = pts.shape[0]
    orig_homog_pts = np.hstack((pts, np.ones((n_total,1), dtype=int)))
    orig_homog_pts_T = orig_homog_pts.T
    prime_homog_pts_T = np.matmul(H, orig_homog_pts_T)
    prime_homog_pts = prime_homog_pts_T.T
    prime_pts = prime_homog_pts * (1.0 / np.tile(prime_homog_pts[:, 2], (3, 1)) ).T
    prime_pts = prime_pts[:,0:2]
    return prime_pts

def map_to_image(H, orig_image, hi=750, wi=750, corners=np.array([[-999],
                                                                  [-999]]),
                                                                  mode='single',
                                                                  pan_scale=1,
                                                                  canvas=None,
                                                                  origin=[0,0]):
    '''maps one whole image to another image, given H'''
    min_x, max_x, min_y, max_y, scale, mapped_image = find_scale(H,
        orig_image, hi, wi, corners, mode, pan_scale, canvas)
    H_inv = np.linalg.pinv(H)
    
    for row_idx, row in enumerate(mapped_image):
        for col_idx, col in enumerate(row):
            # Shift pixels by min if min is less than 0
            if(origin==[0,0]):
                x = np.matmul(H_inv, np.array([[col_idx*scale+min_x],
                                            [row_idx*scale+min_y], [1]]))
            else:
                x = np.matmul(H_inv, np.array([[col_idx*scale+origin[0]],
                                            [row_idx*scale+origin[1]], [1]]))
            x = np.reshape(x, [1, -1]).flatten()
            xcoord = x[0]/x[2]
            ycoord = x[1]/x[2]
            # If outside of source image bounds, leave black
            if (xcoord > 0 and xcoord < orig_image.shape[1] and
                    ycoord > 0 and ycoord < orig_image.shape[0]):
                # use bilinear interpolation which accounts for integers:
                mapped_image[row_idx, col_idx] = bilinear_interpolation(orig_image, xcoord, ycoord)
    # For im.show(), use mapped_image/255
    return mapped_image

def get_pix_value(img, xcoord, ycoord):
    '''Uses weighting to get non-int pixel value'''
    # Check not exceeding bounds of image
    if(math.floor(xcoord+1) >= img.shape[1]):
        high_x = math.floor(xcoord)
    else:
        high_x = math.ceil(xcoord)
    if(math.floor(ycoord+1) >= img.shape[0]):
        high_y = math.floor(ycoord)
    else:
        high_y = math.ceil(ycoord)
    # TODO: check lower boundaries too?
    n1 = [math.ceil(xcoord-1), math.ceil(ycoord-1)]
    n2 = [math.ceil(xcoord-1), high_y]
    n3 = [high_x, math.ceil(ycoord-1)]
    n4 = [high_x, high_y]
    n = np.vstack((n1, np.vstack((n2, np.vstack((n3, n4))))))  # (4,2)
    w = 1/np.sqrt((xcoord - n[:,0])**2 + (ycoord - n[:,1])**2)  # (4,)
    if(len(img.shape) == 3):
        w = np.transpose([w])  # (4,1)
        w = np.tile(w, 3)  # (4,3)
    pn = np.asarray([img[i,j] for i,j in zip(n[:,1], n[:,0])])  # indexing with an array of x,y indices!
    p = np.sum(np.multiply(w, pn), 0) / np.sum(w, 0)
    img[math.floor(ycoord), math.floor(xcoord)]
    return p

def bilinear_interpolation(img, x, y):
    '''Find gray level of point using bilinear interpolation'''
    width = img.shape[1]
    height = img.shape[0]
    A = -1
    B = -1
    C = -1
    D = -1
    # No CD
    if(math.ceil(y) >= height):
        C = 0
        D = 0
        dk = 0
    # No BD
    if(math.ceil(x) >= width):
        B = 0
        D = 0
        dl = 0
    # No AB
    if(math.floor(y) < 0):
        A = 0
        B = 0
        dk = 0
    else:
        dk = y-math.floor(y)
    # No AC
    if(math.floor(x) < 0):
        A = 0
        C = 0
        dl = 0
    else:
        dl = x-math.floor(x)
    # Find remaining valid neighbors
    if(A == -1):
        A = img[(math.floor(y), math.floor(x))]
    if(B == -1):
        B = img[(math.floor(y), math.ceil(x))]
    if(C == -1):
        C = img[(math.ceil(y), math.floor(x))]
    if(D == -1):
        D = img[(math.ceil(y), math.ceil(x))]
    value = (1-dk)*(1-dl)*A + (1-dk)*dl*B + dk*(1-dl)*C + dk*dl*D
    return value

def draw_lines(imgL, imgR, correspondences, scale=1, line_color=(255, 0, 0)):
    '''draw lines connecting imgL to imgR using correspondences'''
    if(imgL.shape[0] < imgR.shape[0]): # pad shorter image
        buffer = np.zeros([imgR.shape[0]-imgL.shape[0], imgL.shape[1]], dtype=np.uint8)
        imgL = np.vstack((imgL, buffer))
    elif(imgR.shape[0] < imgL.shape[0]):
        buffer = np.zeros([imgL.shape[0]-imgR.shape[0], imgR.shape[1]], dtype=np.uint8)
        imgR = np.vstack((imgR, buffer))
    combined_img = np.hstack((imgL, imgR))
    combined_img = cv2.cvtColor(combined_img, cv2.COLOR_GRAY2BGR)
    # Plot pts from the best correspondences individually on the two images first
    imgL_dots = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
    imgR_dots = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)
    for points in correspondences:
        pt1 = np.around(points[0]).astype(int)
        pt2 = np.around(points[1]).astype(int)
        imgL_dots = cv2.circle(imgL_dots, tuple(pt1), radius=math.floor(5/scale), color=(0, 0, 255))
        imgR_dots = cv2.circle(imgR_dots, tuple(pt2), radius=math.floor(5/scale), color=(0, 0, 255))
    display_image('win L', imgL_dots, scale)
    display_image('win R', imgR_dots, scale)
    cv2.destroyAllWindows()
    # Plot lines connecting the two
    for points in correspondences:
        pt1 = np.around(points[0]).astype(int)
        pt2 = np.around([points[1][0] + imgL.shape[1], points[1][1]]).astype(int) # shift x by width of imgL
        combined_img = cv2.line(combined_img, tuple(pt1), tuple(pt2), line_color, math.floor(1/scale))
        combined_img = cv2.circle(combined_img, tuple(pt1), radius=math.floor(5/scale), color=(0, 0, 255))
        combined_img = cv2.circle(combined_img, tuple(pt2), radius=math.floor(5/scale), color=(0, 0, 255))
    display_image('win', combined_img, scale)
    cv2.destroyAllWindows()
    return combined_img


def draw_interest_pts(img, pts, scale=1):
    '''draw circles on img using interest pts/corners'''
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i_point in pts:
        img = cv2.circle(img, tuple(i_point), radius=math.floor(5/scale), color=(0, 0, 255))
    display_image('win', img, scale)
    cv2.destroyAllWindows()
    return img