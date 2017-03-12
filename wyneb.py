# wyneb.py - Wyneb a facial processing library by PiCymru

from skimage import io

import pygame

import math

import dlib

from collections import namedtuple

FaceObject = namedtuple('FaceObject', 'shape rotation')

class Wyneb:
	faceOutline = range(0, 17)
	leftEyebrow = range(18, 22)
	rightEyebrow = range(23, 26)
	nose = range(27, 36)
	leftEye = range(36, 42)
	rightEye = range(43, 48)
	lips = range(49, 68)

	detector = dlib.get_frontal_face_detector()

	def initPredictor(self, filename):
		self.predictor = dlib.shape_predictor(filename)

	def getFaceRotation(self, shapeObj):
	    leftSide = shapeObj.part(self.faceOutline[0])
	    rightSide = shapeObj.part(self.faceOutline[len(self.faceOutline) - 1])

	    ax, ay = (leftSide.x, leftSide.y)
	    bx, by = (rightSide.x, rightSide.y)
	    angleInRads = math.atan2(ay-by, bx-ax)
	    return math.degrees(angleInRads)


	def usePhoto(self, filename):
		self.dlibImage = io.imread(filename)

		self.pygameImage = pygame.image.load(filename)
		# inputImage = pygame.surfarray.make_surface(img)

	def useImage(self, surface):
		self.faceSurface = surface

	def getSurface(self):
		return self.pygameImage

	def findFaces(self):
		faceDataCollection = []

		faces = self.detector(self.dlibImage, 1)

		for k, d in enumerate(faces):
		    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
		        k, d.left(), d.top(), d.right(), d.bottom()))

		    # Get the landmarks/parts for the face in box d.
		    shape = self.predictor(self.dlibImage, d)

		    # Should record:
		    #  - Face landmarks
		    #  - Face bounding box

		    angle = self.getFaceRotation(shape)
		    faceData = FaceObject(shape, angle)
		    faceDataCollection.append(faceData)

		return faceDataCollection

	def getCentralPoint(self, shapeObj, feature):
	    centreX = 0
	    centreY = 0

	    for i in feature:
	        point = shapeObj.part(i)
	        centreX += point.x
	        centreY += point.y

	    centreX /= len(feature)
	    centreY /= len(feature)
	    return (centreX, centreY)

	def getFeatureSize(self, shapeObj, feature):
	    leftX = 99999999
	    rightX = 0
	    topY = 99999999
	    bottomY = 0

	    for i in feature:
	        point = shapeObj.part(i)
	        if point.x < leftX:
	            leftX = point.x

	        if point.x > rightX:
	            rightX = point.x

	        if point.y < topY:
	            topY = point.y

	        if point.y > bottomY:
	            bottomY = point.y

	    width = rightX - leftX
	    height = bottomY - topY
	    return (width, height)

	def outlineFeature(self, shapeObj, feature):
	    for i in feature:
	        point = shapeObj.part(i)
	        print("Part {}: {}".format(i, shapeObj.part(i)))
	        pygame.draw.circle(self.pygameImage, (0, 0, 255), (point.x, point.y), 3, 2)

	def adjustCentre(self, centre, rect):
	    adjustX = centre[0] - (rect.width / 2)
	    adjustY = centre[1] - (rect.height / 2)
	    return (adjustX, adjustY)

	def drawImageOverFeature(self, sourceImage, centrePoint, featureSize, angle):
	    sourceImageRect = sourceImage.get_rect()

	    # scaledSourceImage = pygame.transform.smoothscale(sourceImage, (int(sourceImageRect.width * 0.1), int(sourceImageRect.height * 0.1)))
	    scaledSourceImage = pygame.transform.rotozoom(sourceImage, angle, 0.1)
	    scaledSourceImageRect = scaledSourceImage.get_rect()
	    # scaledSourceImage = pygame.transform.scale(sourceImage, featureSize)
	    
	    adjustedPoint = self.adjustCentre(centrePoint, scaledSourceImageRect)

	    self.pygameImage.blit(scaledSourceImage, adjustedPoint)