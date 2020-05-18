from tkinter import *
import cv2
import numpy as np
from tkinter import messagebox
from PIL import ImageTk, Image
import os
import tkinter.filedialog
import scipy.ndimage as ndi
import poly_point_isect as bot
from skimage import filters
from skimage import data
from skimage import io
from skimage import feature
from skimage.color.adapt_rgb import adapt_rgb,each_channel,hsv_value
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import matplotlib.pyplot as  plt
mainWindow = Tk()
mainWindow.geometry("400x400")
mainWindow.title('MyApp')

framesnumber = 200

def openfile():
    filename = tkinter.filedialog.askopenfilename(title='open')
    return filename


def video_loading():
    path = openfile()
    video_capture = cv2.VideoCapture(path)
    if not video_capture.isOpened():
        raise ValueError("Unable to open this source")

    frames = {}
    for fid in range(0, framesnumber):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = video_capture.read()
        frames[fid] = frame

    medianFrame = MedianFr(frames)
    edges=colorsegmentation(medianFrame)
    hough_line_transform(cv2.imread('canny.png'))



    #courtDetection(medianFrame)
    #colorsegmentation(medianFrame)
    #courtDetection(medianFrame)
    #gray = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
    #courtdet = courtDetection(medianFrame)

    cv2.namedWindow('background', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('background', 730, 580)
    #cv2.imshow('background', medianFrame)
    #cv2.imwrite('background_720x1200.png',medianFrame)

    #movingObjectsDetection(video_capture, medianFrame)

    cv2.namedWindow('line_image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('line_image', 730, 580)

    cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('edges', 730, 580)
    cv2.imshow('edges', edges)

def MedianVal(listVals=[]):
    listVals.sort()
    return listVals[len(listVals) // 2]


def MedianFr(frames):
    height, width = frames[0].shape[:2]
    medianMatrix = np.zeros([height, width, 3], dtype=np.uint8)
    listR = []
    listG = []
    listB = []
    for row in range(0, height):
        for col in range(0, width):
            for frameNr in range(0, framesnumber):
                B = frames[frameNr][row, col][0]
                G = frames[frameNr][row, col][1]
                R = frames[frameNr][row, col][2]
                listR.append(R)
                listG.append(G)
                listB.append(B)
            medianMatrix[row, col][0] = MedianVal(listB)
            medianMatrix[row, col][1] = MedianVal(listG)
            medianMatrix[row, col][2] = MedianVal(listR)

            listR = []
            listG = []
            listB = []
    return medianMatrix


def movingObjectsDetection(cap, medianFrame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    medianFrameGray = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
    height, width = medianFrameGray.shape[:2]
    ret = True
    ct = 0
    while (ret == True and ct < 200):
        ret, frame = cap.read()
        ct += 1
        # Convert current frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference of current frame and the median frame
        differenceframe = cv2.absdiff(frame, medianFrameGray)

        #Treshold to binariza
        th, differenceframe = cv2.threshold(differenceframe,30, 255, cv2.THRESH_BINARY)

        cv2.namedWindow('Motion', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Motion', 730, 580)
        cv2.imshow('Motion', differenceframe)
        cv2.waitKey(40)
    cap.release()

def courtDetection(medianFrame):

    #blur_frame=cv2.GaussianBlur(medianFrame,(21,21),0)
    hsv = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2HSV)
    #lower=[105,84,74] albastru
    #upper=[165,140,125] albastru
    #lower=[130,80,133] roz
    #upper=[195,115,205] roz
    #lower=[150,70,95] mov
    #upper=[210,153,170] mov
    lower=[0,0,168]
    upper=[172,111,255]
    lower=np.array(lower, dtype="uint8")
    upper=np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(medianFrame, medianFrame, mask=mask)
    cv2.imshow('result', result)
    gray=cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    gray2=cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray2, (kernel_size, kernel_size), 0)
    low_threshold = 20
    high_threshold = 180
    kernel=np.ones((5, 5), np.uint8)
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    closing2 = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    closing3=cv2.morphologyEx(closing2, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closing3, kernel ,iterations=1)
    cv2.imshow('dilated_img',dilated)
    cv2.imshow('canny,edges',edges)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 130  # minimum number of pixels making up a line
    max_line_gap = 6  # maximum gap in pixels between connectable line segments
    line_image = np.copy(result) * 0  # creating a blank to draw lines on

    #lines=cv2.HoughLinesP(dilated,1,np.pi/180, 10, minLineLength=120,maxLineGap=5)
    lines = cv2.HoughLinesP(dilated, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    points = []
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                cv2.line(line_image,(x1,y1), (x2,y2), (255,0,0), 3)

    cv2.imshow('linesimg',line_image)
    #cv2.imshow('linesframe',medianFrame)
    #cv2.imshow('edges',dilated)
    cv2.imshow('mask',mask)
    cv2.waitKey(0)
    lines_edges = cv2.addWeighted(result, 0.8, line_image, 1, 0)
    print(lines_edges.shape)
    intersections = bot.isect_segments(points)
    print(intersections)
    for idx, inter in enumerate(intersections):
        a, b = inter
        match = 0
        for other_inter in intersections[idx:]:
            c, d = other_inter
            if abs(c - a) < 8 and abs(d - b) < 8:
                match = 1
                if other_inter in intersections:
                    intersections.remove(other_inter)
                    intersections[idx] = ((c + a) / 2, (d + b) / 2)

        if match == 0:
            intersections.remove(inter)

    for inter in intersections:
        a, b = inter
        for i in range(6):
            for j in range(6):
                lines_edges[int(b) + i, int(a) + j] = [0, 0, 255]

    # Show the result
    cv2.imshow('line_intersections.png', lines_edges)
    cv2.imwrite('line_intersections.png', lines_edges)
    cv2.waitKey(20)

def myrgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.5,0.419,0.081])


def colorsegmentation(medianFrame):
    medianFrame = medianFrame[:, :, ::-1]
    #edges = filters.roberts(rgb2gray(medianFrame))
    edges = feature.canny(rgb2gray(medianFrame))
    #io.imsave("roberts.png",edges_canny)
    #fig,axes=plt.subplots(ncols=2,sharex=TRUE, sharey=TRUE,figsize=(8,4))
    #axes[0].imshow(edges)
    #plt.show()
    return img_as_ubyte(edges)

def hough_line_transform(edges):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 120  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(edges) * 0  # creating a blank to draw lines on
    kernel = np.ones((5, 5), np.uint8)
    #opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    #dilation = cv2.dilate(edges, kernel, iterations=1)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    closing2=cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    closing3=cv2.morphologyEx(closing2, cv2.MORPH_CLOSE, kernel)
    #dilation = cv2.dilate(closing3, kernel, iterations=2)
    cv2.imwrite('closingbun.png',closing3)

    closing_img=cv2.imread('closingbun.png', cv2.IMREAD_GRAYSCALE)
    lines = cv2.HoughLinesP(closing_img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    points = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                cv2.line(line_image, (x1, y1), (x2, y2), (0,255, 0), 5)
                cv2.waitKey(20)
    cv2.imshow('line image',line_image)
    lines_edges = cv2.addWeighted(edges, 1.2,line_image, 1, 0)
    print(lines_edges.shape)
    intersections = bot.isect_segments(points)
    print(intersections)
    for idx, inter in enumerate(intersections):
        a, b = inter
        match = 0
        for other_inter in intersections[idx:]:
            c, d = other_inter
            if abs(c - a) < 8 and abs(d - b) < 8:
                match = 1
                if other_inter in intersections:
                    intersections.remove(other_inter)
                    intersections[idx] = ((c + a) / 2, (d + b) / 2)

        if match == 0:
            intersections.remove(inter)

    for inter in intersections:
        a, b = inter
        for i in range(6):
            for j in range(6):
                lines_edges[int(b) + i, int(a) + j] = [0, 0, 255]

    cv2.imshow('line_intersections.png', lines_edges)
    cv2.imwrite('line_intersections.png', lines_edges)
    cv2.waitKey(0)
    # Show the result

loadButton = Button(mainWindow, text="Load Video", width=15, height=2, command=video_loading).pack()
mainWindow.mainloop()
cv2.destroyAllWindows()
