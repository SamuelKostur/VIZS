#!/usr/bin/env python
import math
import numpy
import cv2

class PathDetector:
    def __init__(self):
        self.initialized = 0

        self.thresh_lower_hsv = []  # lower [H,S,V] boundaries of accepted pathway color
        self.thresh_upper_hsv = []  # upper [H,S,V] boundaries of accepted pathway color

        # parameters for tuning
        self.K = 10  # number of dominant colors to extract in initialization
        self.offset_lower = [30, 40, 30]  # [H,S,V] negative offset
        self.offset_upper = [30, 40, 30]  # [H,S,V] positive offset
        self.field_of_vision_mask_init = False

    def initialize(self, img):
        # Create field of vision mask for find_path_center
        self.field_of_vision_x0 = 0
        self.field_of_vision_y0 = int(img.shape[0]/2)
        self.field_of_vision_mask = self.create_field_of_vision_mask(
            ROWS=img.shape[0], COLS=img.shape[1], 
            lower_width=img.shape[1], upper_width=img.shape[1]/2, height=img.shape[0]/2, 
            x0=self.field_of_vision_x0, y0=self.field_of_vision_y0
        )
        ret, thresh = cv2.threshold(self.field_of_vision_mask, 0, 255, 0)
        self.contours_field_of_vision, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # filtrate image
        img = cv2.medianBlur(img, 3)

        # extract part of the image in the close area of camera
        [sizeY, sizeX, sizeColor] =  img.shape
        centerImg = img[ (sizeY - int(sizeY/9)) : sizeY , int(sizeX/2 - sizeX/4.5) : int(sizeX/2 + sizeX/4.5), :]

        # convert img array from uint8t [y,x,3] to float32t [y*x,3]
        height, width, _ = centerImg.shape
        centerIm = numpy.float32(centerImg.reshape(height * width, 3))

        # find K dominant colors of the extracted area, to determine the color of the pathway
        compactness, labels, centers = cv2.kmeans(K=self.K,
                                                  flags = cv2.KMEANS_RANDOM_CENTERS,
                                                  attempts=1,
                                                  bestLabels=None,
                                                  data=centerIm,
                                                  criteria= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 3, 1.0))


        # convert dominant colors array from float32 [K,3] to uint8t [K,3]
        dom_col_rgb = numpy.uint8(centers)

        # [K,3] matrix of K dominant colors in HSV
        dom_col_hsv = numpy.empty([numpy.size(dom_col_rgb, 0), numpy.size(dom_col_rgb, 1)])

        # [K,3] matrix of accepted lower color boundaries in HSV
        self.thresh_lower_hsv = numpy.empty([numpy.size(dom_col_rgb, 0), numpy.size(dom_col_rgb, 1)])

        # [K,3] matrix of accepted upper color boundaries in HSV
        self.thresh_upper_hsv = numpy.empty([numpy.size(dom_col_rgb, 0), numpy.size(dom_col_rgb, 1)])

        for i in range(numpy.size(dom_col_rgb, 0)):
            # convert color space of corresponding dominant color from BGR to HSV
            dom_col_hsv[i, :] = cv2.cvtColor(numpy.array([[dom_col_rgb[i, :]]]), cv2.COLOR_BGR2HSV).flatten()
            self.thresh_lower_hsv[i, :] = numpy.clip(numpy.subtract(dom_col_hsv[i, :], self.offset_lower), 0, 255)
            self.thresh_upper_hsv[i, :] = numpy.clip(numpy.add(dom_col_hsv[i, :], self.offset_upper), 0, 255)

    def detectPath(self, img):
        mask_comb = self.create_mask(img)

        # filtrate created mask
        mask_filter = cv2.medianBlur(mask_comb,5)
        mask_filter2 = cv2.medianBlur(mask_filter, 19)

        # find pathway border
        #contours, hierarchy = cv2.findContours(mask_filter2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

        img = self.find_path_center(img, mask_comb)

        return img, mask_comb

    def create_mask(self, img):
        # convert BGR img to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_array = []  # array of K masks, one for each dominant color

        for i in range(numpy.size(self.thresh_lower_hsv, 0)):
            # create mask based on accepted color range of corresponding dominant color
            mask_array.append(cv2.inRange(hsv, self.thresh_lower_hsv[i, :], self.thresh_upper_hsv[i, :]))

        # create combined mask from K masks
        mask_comb = mask_array[0]
        for i in range(1, numpy.size(self.thresh_lower_hsv, 0)):
            mask_comb = cv2.bitwise_or(mask_comb, mask_array[i])

        return mask_comb
    
    def create_field_of_vision_mask(self, ROWS, COLS, lower_width, upper_width, height, x0, y0):
        y = lambda x, x0, y0, k : (k*(x - x0) + y0)
        mask = numpy.zeros((ROWS, COLS), dtype='uint8')
    
        k1 = 2*height/(lower_width - upper_width);
        x01 = lower_width + x0;
        y01 = ROWS - 1 - y0;
    
        k2 = -2*height/(lower_width  - upper_width);
        x02 = x0;
        y02 = ROWS - 1 - y0;
    
        for j in range(ROWS):
            for i in range(COLS):
                mask[j][i] = ((j >= y(i, x01, y01, k1)) and j >= y(i, x02, y02, k2) and (j >= y01 - height) and (j <= y01))
        
        return mask

    def find_path_center(self, img, mask_comb):
        saturation = lambda u, lower, upper : min(upper, max(lower, u))
        mask_comb = mask_comb*self.field_of_vision_mask
        
        # calculate centre of the sidewalk
        ROWS = img.shape[0]
        COLS = img.shape[1]
        
        x0 = self.field_of_vision_x0
        y0 = self.field_of_vision_y0
        
        N = 3
        dN = int(saturation((ROWS - y0), 0, ROWS) / N)
        
        x = numpy.zeros(N + 1, dtype=int)
        y = numpy.zeros(N + 1, dtype=int)

        x[0] = int(COLS / 2)
        y[0] = ROWS

        for n in range(N):
            M = cv2.moments(mask_comb[ROWS - y0 - (n + 1) * dN:ROWS - y0 - n * dN, :])
            
            if (M["m00"] == 0):
                x[n + 1] = x[n]
                y[n + 1] = y[n]
            else:
                x[n + 1] = int(M["m10"] / M["m00"])
                y[n + 1] = int(M["m01"] / M["m00"]) + ROWS - y0 - (n + 1) * dN
            
        for n in range(N):
            cv2.line(img, (x[n], y[n]), (x[n + 1], y[n + 1]), (0, 255, 0), 5)

        # M = cv2.moments(mask_comb)
        # x[1] = int(M["m10"] / M["m00"])
        # y[1] = int(M["m01"] / M["m00"])
        # cv2.line(img, (x[0], y[0]), (x[1], y[1]), (0, 255, 0), 5)

        return img

    def process_img(self, img):
        # on first call initialize dominant colors
        if self.initialized == 0:
            self.initialize(img)
        # self.initialized = 1

        # show img with detected path and mask
        imgDet, mask_comb = self.detectPath(img)
        cv2.drawContours(imgDet, self.contours_field_of_vision, -1, (0, 0, 255), 3)
        mask_comb = mask_comb*self.field_of_vision_mask
        cv2.imshow("mask", mask_comb)
        cv2.imshow("path", imgDet)
        cv2.waitKey(1)


def main():
    pathDet = PathDetector()
    
    # img = cv2.imread('Chodnik2.jpg', cv2.IMREAD_COLOR)
    # pathDet.process_img(img)
    # cv2.waitKey(0)
    
    vid = cv2.VideoCapture('robotVid2.mp4')
    
    count = 0
    while vid.isOpened():
        ret, img = vid.read()
        pathDet.process_img(img)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    vid.release()

    # closing all open windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
