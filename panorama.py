# -*- coding: utf-8 -*-
"""
Created on Sat Oct 3

@author: ginac
"""

# %%

import numpy as np
from cv2 import cv2
import lib as lib
import importlib
importlib.reload(lib)
from lib import *
import math
import pickle

# %% Load images

imgs = lib.load_imgs("./original_images/")

# %% Display images

# for img in imgs:
#     lib.display_image('win', img, 1)

# %% Convert to gray

grays = []
for img in imgs:
    grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# %% Display gray images

# for gray in grays:
#     lib.display_image('win', gray, 1)

# %% Find interest points using SIFT

ps = []
dess = []
sift_imgs = []
for i, gray in enumerate(grays):
    p, des, sift_img = lib.use_sift(gray)
    ps.append(p)
    dess.append(des)
    sift_imgs.append(sift_img)

# %% Display interest points

# for sift_img in sift_imgs:
#     display_image('win', sift_img, 1)

# %% Find correspondences using NCC

corrs = []
for i in range(len(sift_imgs)-1):
    corrs.append(lib.find_corr_ssd_or_ncc(grays[i], grays[i+1], ps[i], ps[i+1], dess[i], dess[i+1], 'ncc'))
    print("Done " + str(i+1) + "-" + str(i+2))

# %% Save NCC correspondences to file
with open('nccs.pkl', 'wb') as f:
    pickle.dump(corrs, f)
f.close()

# %% Open NCC correspondences from file

with open('nccs.pkl', 'rb') as f:
    corrs = pickle.load(f)
f.close()

# %% Draw lines for correspondences from NCC method

# for i, corr in enumerate(corrs):
#     lib.draw_lines(grays[i], grays[i+1], corr, 0.5)

# %% RANSAC

n_inliers = []
Hs = []
inlier_corrs = []
outlier_corrs = []
mapped_imgs = []
for i, corr in enumerate(corrs):
    n_inlier, H, inlier_corr, outlier_corr = lib.RANSAC(grays[i], grays[i+1], corrs[i])
    n_inliers.append(n_inlier)
    Hs.append(H)
    inlier_corrs.append(inlier_corr)
    outlier_corrs.append(outlier_corr)
    mapped_imgs.append(lib.map_to_image(H, imgs[i], imgs[i+1]))

# %% Display RANSAC results

# for mapped_img in mapped_imgs:
#     lib.display_image('win', mapped_img/255, 1)

# %% Plot inliers vs outliers

# for i in range(len(inlier_corrs)):
#     lib.draw_lines(grays[i], grays[i+1], np.around(inlier_corrs[i]).astype(int), 0.5, line_color=(0, 255, 0))
#     lib.draw_lines(grays[i], grays[i+1], np.around(outlier_corrs[i]).astype(int), 0.5, line_color=(0, 0, 255))

# %% Refine H with nonlinear least squares

H_LMs = []
for i, inlier_corr in enumerate(inlier_corrs):
    H_LMs.append(lib.refine_H(Hs[i], inlier_corrs[i]))

# %% Compute homographies with respect to the 3rd image

H1_3LM = np.matmul(H_LMs[0], H_LMs[1]) # H1_2 * H2_3
H2_3LM = H_LMs[1] # H2_3
H4_3LM = np.linalg.pinv(H_LMs[2]) # H4_3
H5_3LM = np.matmul(np.linalg.pinv(H_LMs[3]), H4_3LM) # H5_4 * H4_3

# %% Map images with respect to the third image

mapped_img1_3LM = lib.map_to_image(H1_3LM, imgs[0], imgs[2], mode='panorama', pan_scale=3.2)
mapped_img2_3LM = lib.map_to_image(H2_3LM, imgs[1], imgs[2], mode='panorama', pan_scale=3.2)
mapped_img4_3LM = lib.map_to_image(H4_3LM, imgs[3], imgs[2], mode='panorama', pan_scale=3.2)
mapped_img5_3LM = lib.map_to_image(H5_3LM, imgs[4], imgs[2], mode='panorama', pan_scale=3.2)

# %% Display images mapped with respect to the third image

# lib.display_image('win', mapped_img1_3LM/255, 1)
# lib.display_image('win', mapped_img2_3LM/255, 1)
# lib.display_image('win', imgs[2], 1/3)
# lib.display_image('win', mapped_img4_3LM/255, 1)
# lib.display_image('win', mapped_img5_3LM/255, 1)

# %% Create panorama from 5 images with 3rd image as center

# Find min and max y values from first and last image
min_x, max_x1, min_y1, max_y1, scale, mapped_image = lib.find_scale(H1_3LM, grays[0], grays[2], mode='panorama', pan_scale=3.2)
min_x5, max_x, min_y5, max_y5, scale, mapped_image = lib.find_scale(H5_3LM, grays[4], grays[2], mode='panorama', pan_scale=3.2)
min_y = min(min_y1, min_y5)
max_y = max(max_y1, max_y5)

# Create big canvas and identify new origin
pan_scale=3.2
canvas = np.zeros([math.ceil((max_y - min_y)/pan_scale), math.ceil((max_x - min_x)/pan_scale), 3])
origin = [math.ceil(min_x), math.ceil(min_y)]

mapped_img1_LM_panorama = lib.map_to_image(H1_3LM, imgs[0], imgs[2], mode='panorama', pan_scale=pan_scale, canvas=canvas, origin=origin)
print("Done 1")
mapped_img2_LM_panorama = lib.map_to_image(H2_3LM, imgs[1], imgs[2], mode='panorama', pan_scale=pan_scale, canvas=mapped_img1_LM_panorama, origin=origin)
print("Done 2")
Hid = np.identity(3)
mapped_img3_LM_panorama = lib.map_to_image(Hid, imgs[2], imgs[2], mode='panorama', pan_scale=pan_scale, canvas=mapped_img2_LM_panorama, origin=origin)
print("Done 3")
mapped_img4_LM_panorama = lib.map_to_image(H4_3LM, imgs[3], imgs[2], mode='panorama', pan_scale=pan_scale, canvas=mapped_img3_LM_panorama, origin=origin)
print("Done 4")
mapped_img5_LM_panorama = lib.map_to_image(H5_3LM, imgs[4], imgs[2], mode='panorama', pan_scale=pan_scale, canvas=mapped_img4_LM_panorama, origin=origin)
print("Done 5")

# %% Display panorama

# lib.display_image('win', mapped_img1_LM_panorama/255, 1)
# lib.display_image('win', mapped_img2_LM_panorama/255, 1)
# lib.display_image('win', mapped_img3_LM_panorama/255, 1)
# lib.display_image('win', mapped_img4_LM_panorama/255, 1)
lib.display_image('win', mapped_img5_LM_panorama/255, 1)

# %%
