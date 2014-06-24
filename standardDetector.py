import cv2
import numpy as numpy
from operator import itemgetter
import math
from copy import deepcopy
from multiprocessing import Pool, Array, RawArray
import ctypes
import standardDetector
'''
This module implements a function called findStandard which works on cv2 images of 3 colors
findStandard takes one argument (the image) and outputs a projection of the standard. 
It will run fairly slowly for large images at 6MP or higher. If the downsample flag is set to true it will downsample to X pixel width of 2000
It WILL fail when there are more than one standard in the picture.
It is fairly robust to even partially  covered or irregularly shaped standards to a small extent. 
Running this module by itself will find the standard for all jpg files in the same folder and save them to disk in the same folder
'''
HIGH_MEMORY = True #this option enables multiprocessing to speed things up 
MIN_DIST = 10 #beyond this dist two squares are not considered to be the same square
MAX_VECTERROR = 0.97 #cos theta > 0.9 where theta is the angle between the vectors
MAX_NORMALIZED_PERIMETER_ERROR = 0.4
MAX_NUMBER_SQUARES_FROM_MEAN = 4
standardColors = numpy.array(
[[[171, 191,  99],
        [ 41, 161, 229],
        [166, 136,   0],
        [ 50,  50,  50]],

[[176, 129, 130],
        [ 62, 189, 160],
        [150,  84, 188],
        [ 85,  84,  83]],

[[ 65, 108,  90],
        [105,  59,  91],
        [ 22, 200, 238],
        [121, 121, 120]],

[[157, 123,  93],
        [ 98,  84, 195],
        [ 56,  48, 176],
        [161, 161, 160]],

     
[[129, 149, 196],
        [168,  92,  72],
        [ 72, 149,  71],
        [201, 201, 200]],
       

[[ 67,  81, 115],
        [ 45, 123, 220],
        [147,  62,  43],
        [240, 245, 245]]]).astype(numpy.uint8)
labStandardColors = cv2.cvtColor(standardColors, cv2.COLOR_BGR2LAB).astype(numpy.float32)
labRotated = []
for i in range(0,4):
    labRotated.append(numpy.rot90(labStandardColors,-i))
labRotated = numpy.array(labRotated)
class Square:
    def __init__(self):
        pass
    def processContour(self, contour, index, contours, hierarchy):
        self.works = False
        #takes in a contour from the image and processes it into a square
        #if contour is not a square then it will return False, num where num is the number of contours processed
        #furthermore it will set the contour at contours[i] to be False as well. 
        # else it will replace the contour at contours[i] with a Square
        if isinstance(contour,Square):
            return contour, 0
        if isinstance(contour, bool):
            return False, 0
        if cv2.contourArea(contour) < 20*20:
            contours[index] = False
            return False, 1
        for eps in range(2,20):
            contour = cv2.approxPolyDP(contour, float(eps), True)
            contour = cv2.approxPolyDP(contour, float(eps), True)
        if not (cv2.isContourConvex(contour) and len(contour) == 4):
            contours[index] = False
            return False, 1
        points = []
        dists = []
        angles = []
        for i, point in enumerate(contour):
            points.append(point[0])       
        for i, point in enumerate(points):
            dists.append(numpy.linalg.norm(point- points[i-1]))
            a = numpy.arctan2(points[i-1][0]-points[i][0], points[i-1][1]-points[i][1])
            while a < 0:
                a += numpy.pi
            angles.append(a)
        #make sure length of each side is roughly the same
        #make sure angles of each side is roughly parallel
        mean = numpy.mean(dists)
        if not (numpy.abs(angles[2]-angles[0]) < 0.2 and numpy.abs(angles[3]-angles[1]) < 0.2 and abs(dists[0] - (dists[0]+dists[2])/2) + abs(dists[2] - (dists[0]+dists[2])/2) + abs(dists[1] - (dists[1]+dists[3])/2) + abs(dists[3] - (dists[1]+dists[3])/2)< 5.0 and abs((dists[1]+dists[3])/2 - (dists[0]+dists[2])/2)  < 30):
            contours[index] = False
            return False, 1
        parentIndex = hierarchy[0][index][3]
        count = 0
        self.points = points
        self.dists = dists
        self.perimeter = numpy.sum(dists)
        self.angles = angles
        self.contour = deepcopy(contour)
        self.color = (255,255,255)
        self.labColor = None
        self.gridX = 0
        self.gridY = 0
        self.center = numpy.mean(points, axis=0).astype(int)
        self.tupleCenter = (self.center[0], self.center[1])
        while parentIndex > 0:
            if not isinstance(contours[parentIndex], bool) and not isinstance(contours[parentIndex],Square):                
                truth, c = Square().processContour(contours[parentIndex], parentIndex, contours, hierarchy)
                count += c
            if isinstance(contours[parentIndex],Square):
                contours[index] = False
                return False, 1+count
            else:
                parentIndex = hierarchy[0][parentIndex][3]
        #passed all the tests
        # now set to square
        
        contours[index] = self
        self.works = True
        return self, 1+count
def projectAonB(A, B):
    dist = numpy.sqrt(B[1]*B[1] + B[0]*B[0])
    return (A[1]*B[1] + A[0]*B[0]) / dist
colorYield = numpy.zeros((6,4))
def calcDistance((labColor, shape)):
    print labColor
    sharedlabimg = numpy.frombuffer(standardDetector.sharedlabimg_base.get_obj(), dtype=numpy.float32)
    sharedlabimg = sharedlabimg.reshape(shape)
    return numpy.linalg.norm(sharedlabimg-labColor, axis=2)
def initProcess(share):
    standardDetector.sharedlabimg_base = share
def findStandard(img, downSample=False):
    print 'Downsample'
    if downSample:
        
        s = img.shape        
        img = cv2.resize(img, (int(s[1] * 2000 / s[0]), 2000))
        
   
    total = numpy.zeros(img.shape).astype(numpy.uint8)
    
    labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(numpy.float32)
    
    thresh = 200
    margin = 0.2 #black in between line on the standard is roughly 0.2 of the square width
    squares = []
    output = numpy.copy(img)
    #gen circular mask for houghGrid
    a, b = 30,30
    rad = 30
    y,x = numpy.ogrid[-a:61-a, -b:61-b]
    mask = x*x + y*y <= rad*rad
    horizontalOffsets = []
    verticalOffsets = []
    #for calculating orientation
    if HIGH_MEMORY:
        
        size = labimg.size
        print labimg.size, labimg.shape
        sharedlabimg_base = Array(ctypes.c_float, size)
        p = Pool(initializer=initProcess,initargs=(sharedlabimg_base,))
        sharedlabimg = numpy.frombuffer(sharedlabimg_base.get_obj(), dtype=numpy.float32)
        sharedlabimg = sharedlabimg.reshape(labimg.shape)
        print labimg.size, labimg.shape
        print sharedlabimg.size, sharedlabimg.shape
        sharedlabimg[:,:,:]=labimg[:,:,:]
        '''
        #test to see what sharedlabimg looks like after that
        sharedlabimg = numpy.frombuffer(sharedlabimg_base.get_obj(), dtype=numpy.float32)
        sharedlabimg = sharedlabimg.reshape(labimg.shape)
        cv2.imshow('test2', labimg[::6,::6])
        cv2.imshow('test', sharedlabimg[::6,::6])
        if cv2.waitKey(0) == ord('q'):
            quit()
            '''
            
        nMap = p.map(calcDistance, [(c,labimg.shape) for c in labStandardColors.reshape((labStandardColors.shape[0]*labStandardColors.shape[1],labStandardColors.shape[2]))])
        
    #print 'Find squares of each color'
    for r, row in enumerate(standardColors):
        for c, color in enumerate(row):           
            
            labColor = labStandardColors[r][c]
            print 'Calculating color ', labColor
            #print 'Calculate lab distance'
            if not HIGH_MEMORY:
                n = numpy.linalg.norm(labimg-labColor, axis=2)
            else:
                n = nMap[r*len(row)+c]
            n = n * 255 / n.max()
            n = n.astype(numpy.uint8)
            #print 'Threshold'
            #n = cv2.adaptiveThreshold(n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, int(n.shape[1]*0.02) | 1, 6)
            ret, n = cv2.threshold(n, 50, 255, cv2.THRESH_BINARY_INV)
            #print 'Morphology'
            n = cv2.morphologyEx(n, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
            #cv2.imshow(str(i*4+c), cv2.resize(n, dsize=(0,0), fx=0.2, fy=0.2))                
            #print 'Contours'
            contours,h = cv2.findContours(n, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE )
            #sometimes findcontours doesn't reutnr numpy arrays            
            for i, contour in enumerate(contours):
                contours[i] = numpy.array(contour)
            toDraw = []
            indices = []
            #print 'Process contours'
            for i, contour in enumerate(contours):    
                s = Square()
                s, count = s.processContour(contour, i, contours, h)
                if s:
                    contours[i] = s
            curSquares = []            
            for square in contours:
                if isinstance(square, Square):                    
                    square.color = (int(color[0]), int(color[1]), int(color[2]))
                    curSquares.append(square)
            labels = numpy.zeros((img.shape[0], img.shape[1])).astype(numpy.uint8)               
            means = []     
            #print 'Calculate LAB'
            for i in range(0,len(curSquares)):            
                cv2.drawContours(labels, [curSquares[i].contour], -1, i+1, -1)
                roi = cv2.boundingRect(curSquares[i].contour)                                
                mean = cv2.mean(labimg[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] , numpy.array(labels[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] == i+1).astype(numpy.uint8))
                
                curSquares[i].labColor = (mean[0], mean[1], mean[2])
                ##print curSquares[i].labColor, numpy.linalg.norm(curSquares[i].labColor-labColor)
                #cv2.drawContours(total, [curSquares[i].contour], -1, (255,255,255), -1)
                means.append((numpy.linalg.norm(curSquares[i].labColor-labColor), curSquares[i]))
            
            means.sort(key=itemgetter(0), reverse=True)
            colorYield[r][c] += len(means)
            ##print r, c, colorYield[r][c]
            #print 'Add squares, calculate horizontal offsets and vertical offset'
            if len(means) > 0:                                
                for mean in means:
                    square = mean[1]   
                    #check if there is already a square there
                    fail = False
                    for anotherSquare in squares:
                        if numpy.linalg.norm(anotherSquare.center-square.center) < MIN_DIST:
                            #too close!
                            fail = True
                            ##print 'Too close'
                            break
                    if fail:
                        continue
                           
                    
                    squares.append(square)
                    points = mean[1].contour
                    #draw estimated location of colorchecker
                    t = square.center
                    
                    horizontalOffset = ((points[1][0] - points[0][0]) / 2 + (points[2][0] - points[3][0]) / 2) * 1.3
                    verticalOffset = ((points[2][0] - points[1][0]) / 2 + (points[3][0] - points[0][0]) / 2) * 1.3
                    swap = False
                    if len(horizontalOffsets) == 0:
                        swap = abs(horizontalOffset[0]*(1) + horizontalOffset[1]*(0)) < abs(verticalOffset[0]*(1) + verticalOffset[1]*(0))
                        if swap:
                            horizontalOffsets.append(verticalOffset)
                            verticalOffsets.append(horizontalOffset)    
                            horizontalOffset = horizontalOffsets[-1]
                            verticalOffset = verticalOffsets[-1]
                            
                        else:
                            horizontalOffsets.append(horizontalOffset)
                            verticalOffsets.append(verticalOffset)
                        if horizontalOffset[0] < 0:
                            horizontalOffset = -horizontalOffset
                            horizontalOffsets[-1] = -horizontalOffsets[-1]
                        if verticalOffset[1] < 0:
                            verticalOffset = -verticalOffset
                            verticalOffsets[-1] = -verticalOffsets[-1]
                    else:
                        #check to see which one we're closer to, 
                        swap = numpy.abs(horizontalOffset[0]*horizontalOffsets[0][0] + horizontalOffset[1]*horizontalOffsets[0][1]) < numpy.abs(verticalOffset[0]*horizontalOffsets[0][0] + verticalOffset[1]*horizontalOffsets[0][1])
                        ##print horizontalOffset[0]*horizontalOffsets[0][0] + horizontalOffset[1]*horizontalOffsets[0][1], verticalOffset[0]*horizontalOffsets[0][0] + verticalOffset[1]*horizontalOffsets[0][1], horizontalOffsets[0]
                        if swap:
                            horizontalOffsets.append(verticalOffset)
                            verticalOffsets.append(horizontalOffset)                            
                            horizontalOffset = horizontalOffsets[-1]
                            verticalOffset = verticalOffsets[-1]
                        else:
                            horizontalOffsets.append(horizontalOffset)
                            verticalOffsets.append(verticalOffset)    
                        if projectAonB(horizontalOffset, horizontalOffsets[0]) < 0:
                            horizontalOffset = -horizontalOffset
                            horizontalOffsets[-1] = -horizontalOffsets[-1]                        
                        if projectAonB(verticalOffset, verticalOffsets[0]) < 0:
                            verticalOffset = -verticalOffset
                            verticalOffsets[-1] = -verticalOffsets[-1]
            #print 'Done'         
    #calculate estimated location of colorchecker
                    
                
                    
                
    #print 'Calculating offsets'
    horizontalOffsets = numpy.array(horizontalOffsets)
    verticalOffsets = numpy.array(verticalOffsets)
    h, v = numpy.mean(horizontalOffsets, axis=0), numpy.mean(verticalOffsets, axis=0)
    diagonalOffsetDistance = numpy.max(numpy.array([numpy.linalg.norm(h+v), numpy.linalg.norm(v-h)]))
    ##print h,v,diagonalOffsetDistance
    averagePerimeter = numpy.mean(numpy.array([s.perimeter for s in squares]))
    averagePosition = numpy.mean(numpy.array([s.center for s in squares]), axis=0)
    cv2.circle(total, (int(averagePosition[0]), int(averagePosition[1])), 5, (255,128,255), 5)
    meanHO, meanVO = numpy.mean(horizontalOffsets, axis=0), numpy.mean(verticalOffsets, axis=0)
    ##print len(horizontalOffsets), len(verticalOffsets), len(squares)
    a = numpy.array([[horizontalOffsets[count], verticalOffsets[count], squares[count]] for count in range(0,len(squares)) if numpy.dot(horizontalOffsets[count],meanHO) / numpy.linalg.norm(horizontalOffsets[count]) / numpy.linalg.norm(meanHO)  > MAX_VECTERROR and numpy.dot(verticalOffsets[count],meanVO) / numpy.linalg.norm(verticalOffsets[count]) / numpy.linalg.norm(meanVO)  > MAX_VECTERROR and abs(squares[count].perimeter - averagePerimeter) / averagePerimeter < MAX_NORMALIZED_PERIMETER_ERROR and numpy.linalg.norm(averagePosition - squares[count].center) < MAX_NUMBER_SQUARES_FROM_MEAN * diagonalOffsetDistance])       
    if len(a) > 0:
        horizontalOffsets = a[:,0]
        verticalOffsets = a[:,1]
        squares = a[:,2]
        h, v = numpy.mean(horizontalOffsets, axis=0), numpy.mean(verticalOffsets, axis=0)
        ##print h, v
        hx = h[0]
        hy = h[1]
        vx = v[0]
        vy = v[1]
        
        basis = numpy.linalg.inv(numpy.matrix([[hx,vx], [hy,vy]]))
        for square in squares:     
            cv2.circle(total, (square.center[0], square.center[1]), 5, (255,255,255), 5)
            cv2.drawContours(total, [square.contour], -1, square.color, 5)    
            #change basis vectors
            
            target = numpy.matrix([[square.center[0]], [square.center[1]]])
            out = basis * target  
            square.gridX = out.item((0,0))
            square.gridY = out.item((1,0))            
        squares =sorted(squares, key=lambda square: square.gridX*square.gridX+square.gridY*square.gridY)
        offsetX = sorted(squares, key=lambda square:square.gridX)[0].gridX
        offsetY = sorted(squares, key=lambda square:square.gridY)[0].gridY
        maxX = 6
        maxY = 4
        squareDict = {}
        topLeftSquare = None
        topLeft = 24
        topRightSquare = None
        topRight = 24
        bottomLeftSquare = None
        bottomLeft = 24
        bottomRightSquare = None
        bottomRight = 24
        totalGX = 0
        totalGY = 0
        totalXOff = 0
        totalYOff = 0
        for square in squares: 
            ##print square.gridX, square.gridY
            square.gridX -= offsetX
            square.gridY -= offsetY    
        count = 0
        tsquares = None
        residuals = 0
        bestresiduals = 1000000
        besttsquares = None
        bestmaxX = 0
        bestmaxY = 0
        #print 'Find corner squares, residuals and offsets'
        #smart finding of maxX and maxY givest best possible chance of finding fit
        while count < 4:
            minX = numpy.mean([square.gridX for square in squares if square.gridX >= count+0.5  and square.gridX <= count+1.5]) / (count+1)
            minY = numpy.mean([square.gridY for square in squares if square.gridY >= count+0.5 and square.gridY <= count+1.5]) / (count+1)
            count += 1
            if math.isnan(minX) or math.isnan(minY):
                continue
            
            maxX = 0
            maxY = 0
            residuals = 0
            tsquares = deepcopy(squares)
            for square in tsquares:    
                tx = square.gridX
                ty = square.gridY
                residuals += abs(square.gridX/minX-round(square.gridX/minX)) + abs(square.gridY/minY-round(square.gridY/minY))
                square.gridX = round(square.gridX/minX)
                square.gridY = round(square.gridY/minY)                
                gridX = int(square.gridX)
                gridY = int(square.gridY)    
                ##print tx, ty, minX, minY, gridX, gridY                
                if int(square.gridX) > maxX:
                    maxX = int(square.gridX)
                    totalXOff = tx
                if int(square.gridY) > maxY:
                    maxY = int(square.gridY)
                    totalYOff = ty
                if not gridY in squareDict:
                    squareDict[gridY] = {}
                if not gridX in squareDict[gridY]:
                    squareDict[gridY][gridX] = square                
                if 4-gridX + 6-gridY < bottomRight:
                    bottomRight = 4-gridX + 6-gridY
                    bottomRightSquare = square
                if gridX + gridY < topLeft:
                    topLeft = gridX + gridY
                    topLeftSquare = square
                if 4-gridX + gridY < topRight:
                    topRight = 4-gridX + gridY
                    topRightSquare = square
                if gridX + 6-gridY < bottomLeft:
                    bottomLeft = gridX + 6-gridY
                    bottomLeftSquare = square   
            if residuals < bestresiduals and ((maxX < 6 and maxY < 4) or (maxX < 4 and maxY < 6)):
                bestresiduals = residuals
                besttsquares = tsquares
                bestmaxX = maxX
                bestmaxY = maxY
            ##print maxX, maxY, 'max'
        
        #compare to base case
        maxX = 0
        maxY = 0
        residuals = 0
        #print 'Find more residuals'
        tsquares = deepcopy(squares)
        for square in tsquares:    
            tx = square.gridX
            ty = square.gridY
            ##print tx,ty
            residuals += abs(square.gridX-round(square.gridX)) + abs(square.gridY-round(square.gridY))
            square.gridX = round(square.gridX)
            square.gridY = round(square.gridY)
            gridX = int(square.gridX)
            gridY = int(square.gridY)     
            if int(square.gridX) > maxX:
                maxX = int(square.gridX)
                totalXOff = tx
            if int(square.gridY) > maxY:
                maxY = int(square.gridY)
                totalYOff = ty
            if not gridY in squareDict:
                squareDict[gridY] = {}
            if not gridX in squareDict[gridY]:
                squareDict[gridY][gridX] = square                
            if 4-gridX + 6-gridY < bottomRight:
                bottomRight = 4-gridX + 6-gridY
                bottomRightSquare = square
            if gridX + gridY < topLeft:
                topLeft = gridX + gridY
                topLeftSquare = square
            if 4-gridX + gridY < topRight:
                topRight = 4-gridX + gridY
                topRightSquare = square
            if gridX + 6-gridY < bottomLeft:
                bottomLeft = gridX + 6-gridY
                bottomLeftSquare = square   
        if residuals < bestresiduals and ((maxX < 6 and maxY < 4) or (maxX < 4 and maxY < 6)):
            bestresiduals = residuals
            besttsquares = tsquares
            bestmaxX = maxX
            bestmaxY = maxY
        squares = besttsquares    
        maxX = bestmaxX
        maxY = bestmaxY
        #print 'Found final maxX and maxY'      
        if maxX != 0:
            ax = totalXOff / float(maxX)
        else:
            ax = 1
        if maxY != 0:
            ay = totalYOff / float(maxY)
        else:
            ay = 1
        recalculatedHorizontalOffset = ax * h
        recalculatedVerticalOffset = ay * v
        ##print recalculatedHorizontalOffset, recalculatedVerticalOffset
        #connect them all 
        for square in squares:  
            for nsquare in squares:
                if abs(nsquare.gridX-square.gridX)+abs(nsquare.gridY-square.gridY) == 1:
                    cv2.line(total, square.tupleCenter, nsquare.tupleCenter, 255, 5)
        #make fake squares
        #print 'Make fake squares'
        for cy in range(-6,6):
            for cx in range(-6, 6):
                if cy in squareDict and cx in squareDict[cy]:
                    pass
                else:
                    if not cy in squareDict:
                        squareDict[cy] = {}
                    s = Square()                    
                    nearestIndex = sorted([(abs(square.gridX-cx)+abs(square.gridY-cy), i) for i, square in enumerate(squares)], key=itemgetter(0))[0][1]
                    s.center = ((cx-squares[nearestIndex].gridX)*recalculatedHorizontalOffset+(cy-squares[nearestIndex].gridY)*recalculatedVerticalOffset+squares[nearestIndex].center).astype(int)
                    
                    if s.center[0] > 0 and s.center[1] > 0 and s.center[0] < labimg.shape[1] and s.center[1] < labimg.shape[0]:
                        s.labColor = labimg[s.center[1], s.center[0]]
                        squareDict[cy][cx] = s
                        cv2.circle(total, (s.center[0], s.center[1]), 5, (255,255,255), 5)
                        ##print cx,cy, 'make'
        possibilities = [] 
        #print 'Check possibilities'
        for i, rotatedPossible in enumerate(labRotated):
            width = rotatedPossible.shape[1]
            height = rotatedPossible.shape[0]
            tmaxX = maxX
            tmaxY = maxY
            for y in range(0,height-tmaxY):
                for x in range(0,width-tmaxX):
                    terror = 0
                    count = 0
                    for cy in range(0,tmaxY+1):
                        for cx in range(0,tmaxX+1):
                            if cy in squareDict and cx in squareDict[cy]:
                                square = squareDict[cy][cx]
                                labColor = rotatedPossible[y+cy][x+cx]
                                terror += numpy.linalg.norm(square.labColor-labColor)                                   
                                count += 1
                           
                    ##print terror, count, width, height, tmaxX, tmaxY, i, x, y
                    possibilities.append([1/float(count), terror/float(count), (y,x,i)])
        #print 'Find best possibilities'    
        possibilities.sort(key=itemgetter(0,1))        
        rotMatrix = numpy.array([[0,-1],[1,0]])
        if len(possibilities) > 0:
            ans = possibilities[0][2]
            col = numpy.array([[0,0],[0,5],[3,5],[3,0]])
            regPoints = numpy.matrix(numpy.transpose(col))
            ##print numpy.linalg.matrix_power(rotMatrix, ans[2])            
            regPoints = numpy.array(numpy.transpose(numpy.linalg.matrix_power(rotMatrix, ans[2])*regPoints))            
            regPoints[:,0] -= numpy.min(regPoints[:,0])
            regPoints[:,1] -= numpy.min(regPoints[:,1])
            ##print regPoints
            position = squares[0].center
            xoff = squares[0].gridX
            yoff = squares[0].gridY
            for i, regPoint in enumerate(regPoints):
                regPoint -= numpy.array([ans[1], ans[0]])
                regPoint -= numpy.array([xoff, yoff])
                regPoint = ((regPoint[1]*recalculatedVerticalOffset +regPoint[0]*recalculatedHorizontalOffset)+position).astype(int)
                color = (int(standardColors[col[i][1],col[i][0]][0]),int(standardColors[col[i][1],col[i][0]][1]),int(standardColors[col[i][1],col[i][0]][2]))   
                #snap to squares
                snap = []
                for square in squares:
                    if numpy.linalg.norm(square.center-regPoint) < MIN_DIST*5:
                        snap.append(square)
                for cy in squareDict:
                    for cx in squareDict[cy]:
                        square = squareDict[cy][cx]
                        if numpy.linalg.norm(square.center-regPoint) < MIN_DIST*5:
                            snap.append(square) 
                if len(snap) > 0:
                    regPoint = sorted(snap, key = lambda square:numpy.linalg.norm(square.center-regPoint))[0].center
                regPoints[i] = regPoint                        
                cv2.circle(total, (regPoint[0], regPoint[1]), 20, color ,20)
            pt = cv2.getPerspectiveTransform(numpy.array(regPoints).astype(numpy.float32), numpy.array([[50,50], [50,550], [350,550], [350,50]]).astype(numpy.float32))
            
            total = cv2.warpPerspective(img, pt, (400,600))
            
        else:
            print 'NO POSSIBILITIES FOUND', width, height, maxX, maxY, possibilities
            total = cv2.resize(total, dsize=(0,0), fx=0.2, fy=0.2)
        
            
    
    return total
def getRGBFromWarpedImage(img, spotsize=10):
    '''
    assume points are located at 50,150,250,350 etc
    spotsize controls the sample size (square shape)
    '''
    ret = numpy.zeros((6,4,3))
    for y in range(0,6):
        for x in range(0,4):
            py = int((y+0.5)*100)
            px = int((x+0.5)*100)
            n = numpy.mean(numpy.mean(img[py-spotsize:py+spotsize,px-spotsize:px+spotsize], axis=0), axis=0)
            ret[y][x] = n
            
    return ret
if __name__ == '__main__':
    import os
    files = [file for file in os.listdir('.') if file[-4:] == '.jpg']
    for file in files:
        print file
        standard = findStandard(cv2.imread(file), False)
        cv2.imwrite(file[:-4]+'standard.jpg', standard)
        
    
