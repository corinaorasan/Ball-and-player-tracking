from tkinter import *
import itertools
import cv2
import numpy as np
import tkinter.filedialog
import poly_point_isect as bot
from PIL import Image
from skimage import feature
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import imutils
mainWindow = Tk()
mainWindow.geometry("400x400")
mainWindow.title('MyApp')

framesnumber = 200

from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


class Rectangle:
    def intersection(self, other):
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1<x2 and y1<y2:
            return type(self)(x1, y1, x2, y2)
    __and__ = intersection

    def difference(self, other):
        inter = self&other
        if not inter:
            yield self
            return
        xs = {self.x1, self.x2}
        ys = {self.y1, self.y2}
        if self.x1<other.x1<self.x2: xs.add(other.x1)
        if self.x1<other.x2<self.x2: xs.add(other.x2)
        if self.y1<other.y1<self.y2: ys.add(other.y1)
        if self.y1<other.y2<self.y2: ys.add(other.y2)
        for (x1, x2), (y1, y2) in itertools.product(
            pairwise(sorted(xs)), pairwise(sorted(ys))
        ):
            rect = type(self)(x1, y1, x2, y2)
            if rect!=inter:
                yield rect
    __sub__ = difference

    def __init__(self, x1, y1, x2, y2):
        if x1>x2 or y1>y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def __iter__(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2

    def __eq__(self, other):
        return isinstance(other, Rectangle) and tuple(self)==tuple(other)
    def __ne__(self, other):
        return not (self==other)

    def __repr__(self):
        return type(self).__name__+repr(tuple(self))
def pairwise(iterable):
    # https://docs.python.org/dev/library/itertools.html#recipes
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)



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

    start=cv2.getTickCount()
    print('background')

    backgroundModel=createBackgroundModel(frames)
    end=cv2.getTickCount()
    time=(end-start)/cv2.getTickFrequency()
    print (time)
    im_rgb = cv2.cvtColor(backgroundModel, cv2.COLOR_BGR2RGB)
    Image.fromarray(im_rgb).save('bmTest.jpg')

    # backgroundModel=cv2.imread("bm8.jpg")

    coordinates = courtContours(colorsegmentation(backgroundModel))
    start=cv2.getTickCount()
    movingObjectsDetection(video_capture, backgroundModel, coordinates)
    end = cv2.getTickCount()
    time = (end - start) / cv2.getTickFrequency()
    print(time)
def calculateMedian(listVals=[]):
    listVals.sort()
    return listVals[len(listVals) // 2]


def createBackgroundModel(frames):
    height, width = frames[0].shape[:2]
    medianM = np.zeros([height, width, 3], dtype=np.uint8)
    listRed = []
    listGreen = []
    listBlue = []
    for row in range(0, height):
        for col in range(0, width):
            for frameNr in range(0, framesnumber):
                B = frames[frameNr][row, col][0]
                G = frames[frameNr][row, col][1]
                R = frames[frameNr][row, col][2]
                listRed.append(R)
                listGreen.append(G)
                listBlue.append(B)
            medianM[row, col][0] = calculateMedian(listBlue)
            medianM[row, col][1] = calculateMedian(listGreen)
            medianM[row, col][2] = calculateMedian(listRed)

            listBlue = []
            listGreen = []
            listRed = []
    return medianM


def movingObjectsDetection(cap, backgroundModel, coordinates):

    areaOfInterest = Rectangle(coordinates[0], coordinates[1], coordinates[2], coordinates[3])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    backgroundModelGray = cv2.cvtColor(backgroundModel, cv2.COLOR_BGR2GRAY)
    height, width = backgroundModelGray.shape[:2]
    ret = True

    out = cv2.VideoWriter('outputM5.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (int(cap.get(3)), int(cap.get(4))))

    firstRun = True

    bboxes = []
    colors = []
    trackerType = "BOOSTING"
    while ret:
        ret, frame = cap.read()
        if not ret:
            continue
        if firstRun:
            firstRun = False
            # Convert current frame to grayscale
            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate absolute difference of current frame and the bm
            differenceframe = cv2.absdiff(frame2, backgroundModelGray)
            th,differenceframe = cv2.threshold(differenceframe,48, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.dilate(differenceframe, kernel, iterations=1)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]

            multiTracker = cv2.MultiTracker_create()

            # loop over the contours

            for c in cnts:
                # if cv2.contourArea(c) < 450:
                #     continue
                # if the contour is too small, ignore it
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x1, y1, w1, h1) = cv2.boundingRect(c)
                area = Rectangle(x1,y1,x1+w1,y1+h1)
                if(areaOfInterest & area != None):
                    # cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                    bboxes.append((x1, y1, w1, h1))
                    colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
            for bbox in bboxes:
                multiTracker.add(createTrackerByName(trackerType), frame, bbox)
        else:
            # get updated location of objects in subsequent frames
            success, boxes = multiTracker.update(frame)

            # draw tracked objects
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
                if i==0:
                    cv2.putText(frame, 'Player1', p2, cv2.FONT_ITALIC, 0.5, colors[i], 2)
                else:
                    cv2.putText(frame,'Player2', p2, cv2.FONT_ITALIC, 0.5,colors[i], 2)

            # show frame
            out.write(frame)
            cv2.imshow('TrackingPlayers', frame)

            # quit on ESC button
            if cv2.waitKey(10) & 0xFF == 27:  # Esc pressed
                break

    cap.release()
    out.release()
# def courtDetection(backgroundModel):
#
#     blur_frame=cv2.GaussianBlur(backgroundModel,(21,21),0)
#     hsv = cv2.cvtColor(backgroundModel, cv2.COLOR_BGR2HSV)
#     #lower=[105,84,74] albastru
#     #upper=[165,140,125] albastru
#     #lower=[130,80,133] roz
#     #upper=[195,115,205] roz
#     #lower=[150,70,95] mov
#     #upper=[210,153,170] mov
#     lower=[0,0,168]
#     upper=[172,111,255]
#     lower=np.array(lower, dtype="uint8")
#     upper=np.array(upper, dtype="uint8")
#     mask = cv2.inRange(hsv, lower, upper)
#     result = cv2.bitwise_and(backgroundModel, backgroundModel, mask=mask)
#     cv2.imshow('result', result)
#     gray=cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
#     gray2=cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
#     kernel_size = 5
#     blur_gray = cv2.GaussianBlur(gray2, (kernel_size, kernel_size), 0)
#     low_threshold = 20
#     high_threshold = 180
#     kernel=np.ones((5, 5), np.uint8)
#     edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
#     closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#     closing2 = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
#     closing3=cv2.morphologyEx(closing2, cv2.MORPH_CLOSE, kernel)
#     dilated = cv2.dilate(closing3, kernel ,iterations=1)
#     cv2.imshow('dilated_img',dilated)
#     cv2.imshow('canny,edges',edges)
#
#     rho = 1  # distance resolution in pixels of the Hough grid
#     theta = np.pi / 180  # angular resolution in radians of the Hough grid
#     threshold = 10  # minimum number of votes (intersections in Hough grid cell)
#     min_line_length = 130  # minimum number of pixels making up a line
#     max_line_gap = 6  # maximum gap in pixels between connectable line segments
#     line_image = np.copy(result) * 0  # creating a blank to draw lines on
#
#     #lines=cv2.HoughLinesP(dilated,1,np.pi/180, 10, minLineLength=120,maxLineGap=5)
#     lines = cv2.HoughLinesP(dilated, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
#     points = []
#     if lines is not None:
#         for line in lines:
#             for x1,y1,x2,y2 in line:
#                 points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
#                 cv2.line(line_image,(x1,y1), (x2,y2), (255,0,0), 3)
#
#     cv2.imshow('linesimg',line_image)
#     #cv2.imshow('linesframe',medianFrame)
#     #cv2.imshow('edges',dilated)
#     cv2.imshow('mask',mask)
#     cv2.waitKey(0)
#     lines_edges = cv2.addWeighted(result, 0.8, line_image, 1, 0)
#     print(lines_edges.shape)
#     intersections = bot.isect_segments(points)
#     print(intersections)
#     for idx, inter in enumerate(intersections):
#         a, b = inter
#         match = 0
#         for other_inter in intersections[idx:]:
#             c, d = other_inter
#             if abs(c - a) < 8 and abs(d - b) < 8:
#                 match = 1
#                 if other_inter in intersections:
#                     intersections.remove(other_inter)
#                     intersections[idx] = ((c + a) / 2, (d + b) / 2)
#
#         if match == 0:
#             intersections.remove(inter)
#
#     for inter in intersections:
#         a, b = inter
#         for i in range(6):
#             for j in range(6):
#                 lines_edges[int(b) + i, int(a) + j] = [0, 0, 255]
#
#     # Show the result
#     cv2.imshow('line_intersections.png', lines_edges)
#     cv2.imwrite('line_intersections.png', lines_edges)
#     cv2.waitKey(20)

def myrgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.5,0.419,0.081])


def colorsegmentation(backgroundModel):
    backgroundModel=backgroundModel[:, :, ::-1]
    edges = feature.canny(rgb2gray(backgroundModel))
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
    cv2.imwrite('closingfine.png',closing3)

    closing_img=cv2.imread('closingfine.png', cv2.IMREAD_GRAYSCALE)
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

def courtContours(edges):
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(edges, kernel, iterations=1)

    closing_img = dilate
    contours, hierarchy = cv2.findContours(closing_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    import random as rng

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    x, y, w, h = cv2.boundingRect(cnt)
    drawing = np.zeros((closing_img.shape[0], closing_img.shape[1], 3), dtype=np.uint8)
    coordinates = [x-10, y-33, w+x+20, h+y+52]
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        if i == 4:
            color = (100, 255, 33)
            cv2.rectangle(drawing, (x, y), (x + w, y + h), (100, 255, 33), 2)
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    return coordinates

loadButton = Button(mainWindow, text="Load Video", width=15, height=2, command=video_loading).pack()
createModel=Button(mainWindow, text="Create Model", width=15, height=2).pack()
mainWindow.mainloop()
cv2.destroyAllWindows()
