#! /home/ahmed/anaconda3/bin/python

# Libraries
import numpy as np
from skimage import io
from skimage.external import tifffile
import os
import fnmatch

# Extract the names of the files
movies = [movie for movie in os.listdir() if fnmatch.fnmatch(movie, 'movie*')]

# Get the average of each movie
for movie in movies:
    # Read the movie stack
    img = io.imread(movie)
    # Save the mean Z projection of the stack only if it consists of 50 frames
    if img.shape[0] == 50:
        tifffile.imsave('AVG_{}'.format(movie), img.mean(axis = 0))