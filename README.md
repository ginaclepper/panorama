# Panorama
Uses image mosaicing to generate a panorama from five pictures of the same scene.

| ![A city street](https://github.com/ginaclepper/panorama/blob/main/original_images/1.jpg) | 
|:--:| 
| *Original image 1* |

| ![A city street](https://github.com/ginaclepper/panorama/blob/main/original_images/2.jpg) | 
|:--:| 
| *Original image 2* |

| ![A city street](https://github.com/ginaclepper/panorama/blob/main/original_images/3.jpg) | 
|:--:| 
| *Original image 3* |

First, SIFT is used to identify interest points.

| ![Colorful circles of varying scales mark interest points](https://github.com/ginaclepper/panorama/blob/main/results/1%20Sift.png) | 
|:--:| 
| *Interest points for image 1 using SIFT* |

NCC is used to establish correspondences.

| ![Blue lines connect corresponding points in two side-by-side images](https://github.com/ginaclepper/panorama/blob/main/results/1%20to%202%20using%20NCC.png) | 
|:--:| 
| *Correspondences between images 1 and 2 using NCC* |

Then, automated homography estimation is implemented using RANSAC with a linear least squares algorithm.

| ![Green lines connect corresponding points in two side-by-side images](https://github.com/ginaclepper/panorama/blob/main/results/1-2%20inliers.png) | 
|:--:| 
| *Inlier correspondences between images 1 and 2 using RANSAC* |

| ![Red lines connect corresponding points in two side-by-side images](https://github.com/ginaclepper/panorama/blob/main/results/1-2%20outliers.png) | 
|:--:| 
| *Outlier correspondences between images 1 and 2 using RANSAC* |

| ![A distorted image of a city street](https://github.com/ginaclepper/panorama/blob/main/results/1%20to%202%20RANSAC.png) | 
|:--:| 
| *Homography mapping image 1 to 2 using RANSAC* |

The homography is refined using a nonlinear least squares algorithm. Matrix multiplication is used to put all images in the same reference frame (that of the center image).

| ![A distorted image of a city street](https://github.com/ginaclepper/panorama/blob/main/results/1%20to%203%20LM.png) | 
|:--:| 
| *Homography mapping image 1 to 3 using the Levenberg–Marquardt algorithm* |

Finally, all images are plotted on the same canvas to create a panorama.

| ![A panorama of a city street](https://github.com/ginaclepper/panorama/blob/main/results/panorama%20w%20LM.png?raw=true) | 
|:--:| 
| *Final panorama* |
