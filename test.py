#!/usr/bin/python

# Wyneb proof of concept test

import sys
import os
import dlib
import glob
from skimage.draw import circle

import pygame
import pygame.gfxdraw

import wyneb

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then image file to use as the second.\n"
        "Example command line:\n"
        "    ./test.py shape_predictor_68_face_landmarks.dat faces/me.jpg\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_filename = sys.argv[2]

wyneb = wyneb.Wyneb()

wyneb.initPredictor(predictor_path)
# imageWindow = dlib.image_window()

print("Processing file: {}".format(faces_filename))

wyneb.usePhoto(faces_filename)

# inputImage = pygame.surfarray.make_surface(img)
imageRect = wyneb.getSurface().get_rect()
screen = pygame.display.set_mode((imageRect.width, imageRect.height))

faces = wyneb.findFaces()

loveheart = pygame.image.load("overlays/kablam-glossy-heart-800px.png").convert_alpha()

for face in faces:
    print("Face rotation = {}".format(face.rotation))
    leftEyeCentrePoint = wyneb.getCentralPoint(face.shape, wyneb.leftEye)
    rightEyeCentrePoint = wyneb.getCentralPoint(face.shape, wyneb.rightEye)
    eyeSize = wyneb.getFeatureSize(face.shape, wyneb.leftEye)

    wyneb.drawImageOverFeature(loveheart, leftEyeCentrePoint, eyeSize, face.rotation)
    wyneb.drawImageOverFeature(loveheart, rightEyeCentrePoint, eyeSize, face.rotation)

    wyneb.outlineFeature(face.shape, wyneb.nose)
    wyneb.outlineFeature(face.shape, wyneb.faceOutline)

screen.blit(wyneb.getSurface(), imageRect)

pygame.display.flip()

dlib.hit_enter_to_continue()
