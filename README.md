# Panorama
Uses image mosaicing to generate a panorama from multiple pictures of the same scene.

First, SIFT is used to identify interest points. NCC is used to establish correspondences. Then, automated homography estimation is implemented using RANSAC with a linear least squares algorithm. The homography is refined using a nonlinear least squares algorithm. Finally, all 5 images are put in the same reference frame and plotted together to create a panorama.
