# IMPORTS
import numpy as np
import imageio
import matplotlib.pyplot as plt
import skimage.color as color
import skimage.filters as filters
import skimage.measure as measure
from skimage.util import random_noise
from skimage.segmentation import slic
from skimage.feature import canny
import skimage.feature as feature
import skimage.transform as transform
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import rotate
from prettytable import PrettyTable

# Reading in images:
avengers = imageio.imread('data/image_data/avengers_imdb.jpg')
bush_house = imageio.imread('data/image_data/bush_house_wikipedia.jpg')
forestry = imageio.imread('data/image_data/forestry_commission_gov_uk.jpg')
rolland = imageio.imread('data/image_data/rolland_garros_tv5monde.jpg')

# 1.1 
# Determining the size of the avengers_imdb.jpg image
avengers_shape = avengers.shape

table_1 = PrettyTable()
table_1.title = 'Image Mining 1.1'
table_1.field_names = ["Question", "Findings"]
table_1.add_row(['Size of avengers_imdb.jpg:',  avengers_shape])
table_1.align = "l"
print(table_1)

plt.figure(figsize=(10,5))
plt.subplot(131)
plt.title("Original image", fontsize = 8)
plt.axis('off')
plt.imshow(avengers)

# Produce a grayscale representation
avengers_grayscale = color.rgb2gray(avengers)

plt.subplot(132)
plt.title("Grayscale image", fontsize = 8)
plt.axis('off')
plt.imshow(avengers_grayscale, cmap=plt.cm.gray)

# Produce a binary representation
threshold = filters.threshold_otsu(avengers_grayscale)
avengers_binary = avengers_grayscale > threshold

plt.subplot(133)
plt.title("Black-and-white image", fontsize = 8)
plt.axis('off')
plt.imshow(avengers_binary, cmap=plt.cm.gray)
plt.savefig('outputs/avengers_all.jpg')

# 1.2
plt.figure(figsize=(10,3))

plt.subplot(141)
plt.title("Original image", fontsize = 8)
plt.axis('off')
plt.imshow(bush_house)

# Add Gaussian random noise to the bush_house_wikipedia.jpg image with variance 0.1
bush_house_rn = random_noise(bush_house, mode='gaussian', var=0.1)

plt.subplot(142)
plt.title("Gaussian random noise", fontsize = 8)
plt.axis('off')
plt.imshow(bush_house_rn)

# Filtering the perturbed image with a Gaussian mask sigma = 1
bush_house_gaus_mask = gaussian_filter(bush_house_rn, sigma=1)

plt.subplot(143)
plt.title("Gaussian mask", fontsize = 8)
plt.axis('off')
plt.imshow(bush_house_gaus_mask)

# Applying a uniform smoothing mask (9x9)
bush_house_uniform = uniform_filter(bush_house_gaus_mask, size = 9)

plt.subplot(144)
plt.title("Uniform smoothing mask", fontsize = 8)
plt.axis('off')
plt.imshow(bush_house_uniform)
plt.savefig('outputs/bush_house_all.jpg')

# 1.3
# Divide the forest_commission_gov_uk.jpg into 5 segments using k-means segmentation
forest_segmented = slic(forestry, n_segments=5, compactness=20.0, start_label=1)

plt.figure(figsize=(10,3))

plt.subplot(121)
plt.title("Original image", fontsize = 8)
plt.axis('off')
plt.imshow(forestry)

plt.subplot(122)
plt.title("Segmented image", fontsize = 8)
plt.axis('off')
plt.imshow(forest_segmented)
plt.savefig('outputs/forestry_all.jpg')

# 1.4 
#  Perform Canny edge detection and apply Hough transform on rolland_garros_tv5monde.jpg
rolland_grayscale = color.rgb2gray(rolland)
rolland_canny = canny(rolland_grayscale)
lines = transform.probabilistic_hough_line(np.fliplr(rotate(rolland_canny, 180)), threshold=160, line_length=105, line_gap=10)

plt.figure(figsize=(10,3))

plt.subplot(131)
plt.title("Original image", fontsize = 8)
plt.axis('off')
plt.imshow(rolland)

plt.subplot(132)
plt.title("Canny edge detection", fontsize = 8)
plt.axis('off')
plt.imshow(rolland_canny)

plt.subplot(133)
plt.title("Hough transform", fontsize = 8)
plt.axis('off')
for line in lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
plt.savefig('outputs/rolland_all.jpg')