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

# Reading in images:
avengers = imageio.imread('data/image_data/avengers_imdb.jpg')
bush_house = imageio.imread('data/image_data/bush_house_wikipedia.jpg')
forestry = imageio.imread('data/image_data/forestry_commission_gov_uk.jpg')
rolland = imageio.imread('data/image_data/rolland_garros_tv5monde.jpg')

# 1.1 
# Determining the size of the avengers_imdb.jpg image
print(avengers.shape)

# Produce a grayscale representation
avengers_grayscale = color.rgb2gray(avengers)
plt.axis('off')
plt.imshow(avengers_grayscale, cmap=plt.cm.gray)
plt.savefig('outputs/avengers_grayscale.jpg')

# Produce a binary representation
threshold = filters.threshold_otsu(avengers_grayscale)
print('Otsu method threshold = ', threshold)
avengers_binary = avengers_grayscale > threshold
plt.imshow(avengers_binary, cmap=plt.cm.gray)
plt.savefig('outputs/avengers_binary.jpg')

# 1.2
# Add Gaussian random noise to the bush_house_wikipedia.jpg image with variance 0.1
bush_house_rn = random_noise(bush_house, mode='gaussian', var=0.1)

# Filtering the perturbed image with a Gaussian mask sigma = 1
bush_house_gaus_mask = gaussian_filter(bush_house_rn, sigma=1)
plt.imshow(bush_house_gaus_mask)
plt.savefig('outputs/bush_house_gaus_mask.jpg')


# Applying a uniform smoothing mask (9x9)
bush_house_uniform = uniform_filter(bush_house_gaus_mask, size = 9)
plt.imshow(bush_house_uniform)
plt.savefig('outputs/bush_house_uniform.jpg')

# 1.3
# Divide the forest_commission_gov_uk.jpg into 5 segments using k-means segmentation
forest_segmented = slic(forestry, n_segments=5, compactness=20.0, start_label=1)
plt.imshow(forest_segmented)
plt.savefig('outputs/forest_segmented.jpg')

# 1.4 
#  Perform Canny edge detection and apply Hough transform on rolland_garros_tv5monde.jpg
rolland_grayscale = color.rgb2gray(rolland)
rolland_canny = canny(rolland_grayscale)
plt.imshow(rolland_canny)

lines = transform.probabilistic_hough_line(np.fliplr(rotate(rolland_canny, 180)), threshold=160, line_length=105, line_gap=10)
plt.figure()
for line in lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
plt.savefig('outputs/probabilistic_hough_line.jpg')

