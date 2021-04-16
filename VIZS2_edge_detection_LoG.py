# -*- coding: utf-8 -*-

import math
from scipy.integrate import nquad
from numpy import zeros, round
import cv2
from numba import jit

@jit
def conv2(f, g, shape = 'full'):
    # 2D konvolucia
    #
    # Vstupy:
    #       f     ... Obraz (2D numpy.array)
    #       g     ... Kernel (2D numpy.array)
    #       shape ... Typ konovlucie :
    #                       'full'  - Plna konvolucia, zvatsi obraz
    #                       'same'  - Konvolucia zachova rozmery obrazu
    #                       'valid' - Konvolucia obmedzena velkostou kernelu, zmensi obraz
    #
    # Vystup:
    #       h ... Obraz po konvolucii (2D numpy.arrray)
    
    row_f, col_f = f.shape
    row_g, col_g = g.shape

    # Urcenie typu konvolucie
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

    # Prechod obrazom
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

def zero_crossings(arr, VAL=1):
    # Detekuje prechody 0 => hrany
    #
    # Vstupy:
    #       arr ... Obraz v odtienoch sivej (2D numpy.arrray)
    #       VAL ... Vystupna hodnota pri detekovani hrany
    #
    # Vystup:
    #       out ... Detekovane hrany (Binarne 2D numpy.arrray)
    
    ROW, COL = arr.shape
    out = zeros((ROW, COL))
    
    # Prechod obrazom
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
    # Vytvorenie stvorcoveho LoG kernelu
    #
    # Vstupy:
    #       size ... Rozmery kernelu
    #       sig  ... Rozptyl Gaussovej funkcie
    #
    # Vystup:
    #       f ... LoG operator (2D numpy.array)
    
    # Inicializacia
    f = zeros((size, size))
    p = (size - 1)/2
    
    # Generovanie pre size x size buniek
    for m in range(size):
        yspan = [m - p - 0.5, m - p + 0.5]
        for n in range(size):
            xspan = [n - p - 0.5, n - p + 0.5]
            # 2D integracia
            f[m, n], _ = nquad(LoG, [xspan, yspan], args = (sig,))
     
    return f

def LoG(x, y, sig):
    # Matematicka funkcia LoG kernelu
    #
    # Vstupy:
    #       x   ... x-suradnica
    #       y   ... y-suradnica
    #       sig ... Rozptyl
    #
    # Vystup:
    #       Hodnota v danom bode [x, y]
    
    return ((x**2 + y**2)/(2*sig**2) - 1)/(math.pi*sig**4) * math.exp(-(x**2 + y**2)/(2*sig**2))


def main():
    # Nacitanie obrazu
    img = cv2.imread('image.jpeg')
    
    # Konverzia farebneho obrazu na obraz v odtienoch sivej
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    
    # Vygenerovanie LoG kernelu
    size = 5        # Velkost kernelu
    sig = 1         # Rozptyl Gaussovej funkcie
    scale = 526     # Skalovanie int kernelu
    
    # float64 kernel:
    kernel = LoG_kernel(size, sig)
    # int64 kernel:
    # kernel = round(scale*LoG_kernel(size, sig)).astype('int64')
    
    # Vypocet konovlucie obrazu a LoG operatora
    output = conv2(img, kernel, 'same')
    
    # Detekcia hran (prechody 0)
    edges = zero_crossings(output, 255)
    
    # Vykreslenie obrazu
    cv2.imshow('Image', img)
    cv2.imshow('LoG', output)
    cv2.imshow('Edges', edges)
    
    # Ukoncenie
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if  __name__ == '__main__':
    main()