import numpy as np
from scipy.misc import imsave
import math

def visualize(images, filename):
  assert images.max() <= 255 and images.min() >= 0
  (B, H, W, C) = images.shape
  length = int(math.floor(math.sqrt(B)))
  result = np.zeros((length * H, length * W, C), dtype=np.uint8)
  for i in xrange(length):
    for j in xrange(length):
      result[i * H: (i + 1) * H, j * W: (j + 1) * W] = images[i * length + j]

  imsave(filename, result.squeeze())
     
