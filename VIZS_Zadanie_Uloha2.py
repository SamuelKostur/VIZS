# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:41:58 2021

@author: Asus
"""

import time
import math
from scipy.integrate import quad, nquad
from numpy import zeros
import cv2
from numba import jit

@jit
def conv(f, g):
    f = f.reshape(-1,)
    g = g.reshape(-1,)
    len_f = len(f)
    len_g = len(g)
    len_h = len_f + len_g - 1
    h = zeros((len_h,))
    
    for m in range(len_h):
        for n in range(max(0, m + 1 - len_f), min(m + 1, len_g)):
            h[m] = h[m] + f[m - n]*g[n]
    
    return h

@jit
def conv2(f, g, shape = 'full'):
    row_f, col_f = f.shape
    row_g, col_g = g.shape

    if (shape == 'full'):
        MIN_ROW = 0
        MIN_COL = 0
    elif (shape == 'same'):
        MIN_ROW  = int((row_g - 1)/2)
        MIN_COL = int((col_g - 1)/2)
    elif (shape == 'valid'):
        MIN_ROW  = row_g - 1
        MIN_COL = col_g - 1
    else:
        return None
    
    MAX_ROW = row_f + row_g - 1 - MIN_ROW
    MAX_COL = col_f + col_g - 1 - MIN_COL
    h = zeros((MAX_ROW - MIN_ROW, MAX_COL - MIN_COL))

    for m in range(MIN_ROW, MAX_ROW):
        k = m - MIN_ROW
        min_row = max(0, m + 1 - row_g)
        max_row = min(m + 1, row_f)
        
        for n in range(MIN_COL, MAX_COL):
            l = n - MIN_COL
            min_col = max(0, n + 1 - col_g)
            max_col = min(n + 1, col_f)
            
            for i in range(min_row, max_row):
                for j in range(min_col, max_col):
                    h[k, l] = h[k, l] + f[i, j]*g[m - i, n - j]
    
    return h

def zero_crossings(arr):
    ROW, COL = arr.shape
    out = zeros((ROW, COL))
    VAL = 1
    
    for m in range(ROW):
        for n in range(COL):
            if (out.item(m, n) == VAL):
                continue
            
            act_val = arr.item(m, n)
            
            if (m < ROW - 1):
                next_row_val = arr.item(m + 1, n)
            else:
                next_row_val = act_val
            
            if (n < COL - 1):
                next_col_val = arr.item(m, n + 1)
            else:
                next_col_val = act_val
            
            if (act_val*next_row_val <= 0):
                if (abs(act_val) <= abs(next_row_val)):
                    out[m, n] = VAL
                elif (m < ROW - 1):
                    out[m + 1, n] = VAL
            
            if (act_val*next_col_val <= 0):
                if (abs(act_val) <= abs(next_col_val)):
                    out[m, n] = VAL
                elif (n < COL - 1):
                    out[m, n + 1] = VAL
    
    return out

def LoG_kernel(size, sig):
    f = zeros((size, size))
    p = (size - 1)/2
    
    for m in range(size):
        yspan = [m - p - 0.5, m - p + 0.5]
        for n in range(size):
            xspan = [n - p - 0.5, n - p + 0.5]
            f[m, n], _ = nquad(LoG, [xspan, yspan], args = (sig,))
     
    return f

def gauss1D_kernel(size, sig):
    f = zeros((size,))
    p = (size - 1)/2
    
    for m in range(size):
        xspan = [m - p - 0.5, m - p + 0.5]
        f[m], _ = quad(gauss, xspan[0], xspan[1], args = (sig,))
     
    return f

def gauss2D_kernel(size, sig):
    f = zeros((size, size))
    p = (size - 1)/2
    
    for m in range(size):
        yspan = [m - p - 0.5, m - p + 0.5]
        for n in range(size):
            xspan = [n - p - 0.5, n - p + 0.5]
            f[m, n], _ = nquad(gauss2D, [xspan, yspan], args = (sig,))
     
    return f

def gauss(x, sig):
    return 1/(math.sqrt(2*math.pi)*sig) * math.exp(-x**2/(2*sig**2))

def gauss2D(x, y, sig):
    return 1/(2*math.pi*sig**2) * math.exp(-(x**2 + y**2)/(2*sig**2))

def LoG(x, y, sig):
    return ((x**2 + y**2)/(2*sig**2) - 1)/(math.pi*sig**4) * math.exp(-(x**2 + y**2)/(2*sig**2))

image = cv2.imread('Image.jpeg')
image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)
kernel = (526*LoG_kernel(9, 1.4)).astype('int32')
# kernel = array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

tic = time.time()
output = conv2(image, kernel, 'same')
edges = 255*zero_crossings(output)
toc = time.time()
elapsed = toc - tic

cv2.imshow('image', image)
cv2.imshow('LoG', output)
cv2.imshow('edges', edges)

cv2.imwrite('output.jpg', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()