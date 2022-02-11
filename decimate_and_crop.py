#!/usr/bin/env python
# coding: utf-8

# Definition of the decimation mask used to reach a given image size from the
# original full size correlator ouput matrices, which size is [h, w] = [81, 89].
# Decimation is done from the position of the correlation peak on the I channel,
# so that it remains unchanged. The reason for this is that this specific point
# will always exist in a real receiver, it is the Prompt point. The final image
# size depends on the cropping method:
# 1. Crop first to [80, 80], then decimate.
# 2. Decimate directly from [81, 89], then crop to obtain a square image.

import sys
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def get_indices(image_size=(40, 40)):
  if image_size == (81, 81): # Decimation factor = 1, just crop, N = min([h, w])
    i_d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 78, 79, 80]
    j_d = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,78, 79, 80, 81, 82,
    83, 84, 85, 86, 87, 88]
    i_Doppler = 40
    j_delay = 23
  elif image_size == (80, 80): # Decimation factor = 1, just crop, N = 80
    i_d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 78, 79]
    j_d = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
    83, 84, 85, 86, 87, 88]
    i_Doppler = 40
    j_delay = 22
  elif image_size == (40, 40): # Decimation factor = 2,
    # N = floor(80/factor) and N = min(floor([h, w]/factor))
    i_d = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
    36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72,
    74, 76, 78]
    j_d = [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
    43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79,
    81, 83, 85, 87]
    i_Doppler = 20
    j_delay = 11
  elif image_size == (27, 27): # 3, N = min(floor([h, w]/factor))
    i_d = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52,
    55, 58, 61, 64, 67, 70, 73, 76, 79]
    j_d = [10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49,52, 55, 58,
    61, 64, 67, 70, 73, 76, 79, 82, 85, 88]
    i_Doppler = 13
    j_delay = 7    
  elif image_size == (26, 26): # 3, N = floor(80/factor)
    i_d = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52,
    55, 58, 61, 64, 67, 70, 73, 76]
    j_d = [13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52,55, 58, 61,
    64, 67, 70, 73, 76, 79, 82, 85, 88]
    i_Doppler = 13
    j_delay = 6
  elif image_size == (20, 20): # 4,
    # N = floor(80/factor) and N = min(floor([h, w]/factor))
    i_d = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68,
    72, 76]
    j_d = [11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63,67, 71, 75,
    79, 83, 87]
    i_Doppler = 10
    j_delay = 5
  elif image_size == (16, 16): # 5,
    # N = floor(80/factor) and N = min(floor([h, w]/factor))
    i_d = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    j_d = [11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76,81, 86]
    i_Doppler = 8
    j_delay = 4
  elif image_size == (13, 13): # 6,
    # N = floor(80/factor) and N = min(floor([h, w]/factor))
    i_d = [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76]
    j_d = [13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85]
    i_Doppler = 6
    j_delay = 3
  elif image_size == (11, 11): # 7,
    # N = floor(80/factor) and N = min(floor([h, w]/factor))
    i_d = [5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75]
    j_d = [17, 24, 31, 38, 45, 52, 59, 66, 73, 80, 87]
    i_Doppler = 5
    j_delay = 2
  elif image_size == (10, 10): # 8,
    # N = floor(80/factor) and N = min(floor([h, w]/factor))
    i_d = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
    j_d = [15, 23, 31, 39, 47, 55, 63, 71, 79, 87]
    i_Doppler = 5
    j_delay = 2
  elif image_size == (9, 9): # 9, N = min(floor([h, w]/factor))
    i_d = [4, 13, 22, 31, 40, 49, 58, 67, 76]
    j_d = [13, 22, 31, 40, 49, 58, 67, 76, 85]
    i_Doppler = 4
    j_delay = 2
  # elif image_size == (8, 8): # 9, N = floor(80/factor)
    # i_d = [4, 13, 22, 31, 40, 49, 58, 67]
    # j_d = [22, 31, 40, 49, 58, 67, 76, 85]
    # i_Doppler = 4
    # j_delay = 1
  elif image_size == (8, 8): # 10,
    # N = floor(80/factor) and N = min(floor([h, w]/factor))
    i_d = [0, 10, 20, 30, 40, 50, 60, 70]
    j_d = [11, 21, 31, 41, 51, 61, 71, 81]
    i_Doppler = 4
    j_delay = 2
  elif image_size == (7, 7): # 11,
    # N = floor(80/factor) and N = min(floor([h, w]/factor))
    i_d = [7, 18, 29, 40, 51, 62, 73]
    j_d = [20, 31, 42, 53, 64, 75, 86]
    i_Doppler = 3
    j_delay = 1
  elif image_size == (6, 6): # 12,
    # N = floor(80/factor) and N = min(floor([h, w]/factor))
    i_d = [4, 16, 28, 40, 52, 64]
    j_d = [19, 31, 43, 55, 67, 79]
    i_Doppler = 3
    j_delay = 1
  # elif image_size == (6, 6): # 13,
    # N = floor(80/factor) and N = min(floor([h, w]/factor))
    # i_d = [1, 14, 27, 40, 53, 66]
    # j_d = [18, 31, 44, 57, 70, 83]
    # i_Doppler = 3
    # j_delay = 1
  elif image_size == (5, 5): # 14,
    # N = floor(80/factor) and N = min(floor([h, w]/factor))
    i_d = [12, 26, 40, 54, 68]
    j_d = [31, 45, 59, 73, 87]
    i_Doppler = 2
    j_delay = 0
  else:
    logger.error("Wrong image size specification: {:d}x{:d}".\
      format(image_size[0], image_size[1]))
    sys.exit(1)
  return (i_d, j_d, i_Doppler, j_delay)

