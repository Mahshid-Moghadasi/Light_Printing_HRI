from FamousCurves import *
from machinaRobot import *
import sys
import re
import math
import time
import websockets
import asyncio
import torch
import cv2
import time
import argparse
import posenet
import Int_exp01_cam
import math
import numpy as np


def distance(pointA, pointB):
    d = (pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2 + (pointA[2] - pointB[2]) ** 2
    d = math.sqrt(d)
    return d


def remap(x,xmin, xmax,targetMin,targetMax):
    if x<=xmin: return targetMin
    if x>=xmax: return targetMax
    return (x-xmin) / (xmax-xmin) * (targetMax-targetMin) + targetMin


def createPointinRobotSpace(y,z):
    y = y*20
    z = z*20
    z = z+400
    return y,z

def normalizeVec(vec):
    return vec/np.linalg.norm(vec)


###########################################################################################################################
def checkX(x):
    if x<200:
        return 200
    elif x>500:
        return 500
    else:
        return x

def checkY(y):
    if y<-450:
        return -450
    elif y>450:
        return 450
    else:
        return y

def checkZ(z):
    if z<160:
        return 160
    elif z>550:
        return 550
    else:
        return z
############################################################################################################################
# Camera Settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125) #default=0.7125)
args = parser.parse_args()

pose_dictionary = {'nose': [0,0],
                   'leftEye':[0,0],
                   'rightEye':[0,0],
                   'leftEar':[0,0],
                   'rightEar':[0,0],
                   'leftShoulder':[0,0],
                   'rightShoulder':[0,0],
                   'leftElbow':[0,0],
                   'rightElbow':[0,0],
                   'leftWrist':[0,0],
                   'rightWrist':[0,0],
                   'leftHip' : [0,0],
                   'rightHip': [0,0],
                   'leftKnee': [0,0],
                   'rightKnee': [0,0],
                   'leftAnkle': [0,0],
                   'rightAnkle': [0,0]
                   }

############################################################################################################################
# Context point and robot initial settings
start_point = (465, 124.56, 272.79, -1, 1, 0, 0, 1, 0)  # rotating the sixth axis: (300, 0, 300, -1, this one, 0, 0, 1, 0) It's good practice to go between -10 and 10
context_points = [(366.07,147.61,223.36, -1, 1, 0, 0, 1, 0)]

# 250 points - should be between 200 and 500 - here we set the bound between 465 and 300
context_points_x = [465,464.34,463.68,463.02,462.36,461.7,461.04,460.38,459.72,459.06,458.4,457.74,457.08,456.42,455.76,455.1,454.44,453.78,453.12,452.46,451.8,451.14,450.48,449.82,449.16,448.5,447.84,447.18,446.52,445.86,445.2,444.54,443.88,443.22,442.56,441.9,441.24,440.58,439.92,439.26,438.6,437.94,437.28,436.62,435.96,435.3,434.64,433.98,433.32,432.66,432,431.34,430.68,430.02,429.36,428.7,428.04,427.38,426.72,426.06,425.4,424.74,424.08,423.42,422.76,422.1,421.44,420.78,420.12,419.46,418.8,418.14,417.48,416.82,416.16,415.5,414.84,414.18,413.52,412.86,412.2,411.54,410.88,410.22,409.56,408.9,408.24,407.58,406.92,406.26,405.6,404.94,404.28,403.62,402.96,402.3,401.64,400.98,400.32,
            399.66,399,398.34,397.68,397.02,396.36,395.7,395.04,394.38,393.72,393.06,392.4,391.74,391.08,390.42,389.76,389.1,388.44,387.78,387.12,386.46,385.8,385.14,384.48,383.82,383.16,382.5,381.84,381.18,380.52,379.86,379.2,378.54,377.88,377.22,376.56,375.9,375.24,374.58,373.92,373.26,372.6,371.94,371.28,370.62,369.96,369.3,368.64,367.98,367.32,366.66,366,365.34,364.68,364.02,363.36,362.7,362.04,361.38,360.72,360.06,359.4,358.74,358.08,357.42,356.76,356.1,355.44,354.78,354.12,353.46,352.8,352.14,351.48,350.82,350.16,349.5,348.84,348.18,347.52,346.86,346.2,345.54,344.88,344.22,343.56,342.9,342.24,341.58,340.92,340.26,339.6,338.94,338.28,337.62,336.96,336.3,335.64,334.98,
            334.32,333.66,333,332.34,331.68,331.02,330.36,329.7,329.04,328.38,327.72,327.06,326.4,325.74,325.08,324.42,323.76,323.1,322.44,321.78,321.12,320.46,319.8,319.14,318.48,317.82,317.16,316.5,315.84,315.18,314.52,313.86,313.2,312.54,311.88,311.22,310.56,309.9,309.24,308.58,307.92,307.26,306.6,305.94,305.28,304.62,303.96,303.3,302.64,301.98,301.32,300.66,300]

targetMin = 0
targetMax = math.pi * 2.0
xmin = 300
xmax = 465

speed = 100
bot = MachinaRobot()
start = time.time()
while_var = True

############################################################################################################################
# Interaction settings
###########  Hypocycloid 2 #########################
# a should be between 1.00 and 8.0
# b should be between 0.3 and 1.0
# c should be between 0.1 and 7.0
piN = 4
t_values = createT(0,math.pi*piN,250)
a = 5.08
b = 0.38
c = 7.0
scale = 20.0

a_min, a_max = 2.0, 8.0
b_min, b_max = 0.3 , 1.0
c_min, c_max = 0.1 , 7.0


########  run these in the beginning  ################################################
# bot.AxesTo(0, 0, 0, 0, 90, 0)
# bot.TransformTo(start_point)

async def feedback():
    address = "ws://127.0.0.1:6999/Bridge"
    async with websockets.connect(address) as websocket:
        f = await websocket.recv()
        return f


if __name__ == '__main__':
    try:
        # camera settings
        cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        model = posenet.load_model(args.model)
        model = model.cuda()
        output_stride = model.output_stride

        # starting move
        bot.SpeedTo(speed)
        t_next = remap(context_points_x[0],xmin, xmax,targetMin,targetMax)
        next_point_yz = hypocycloid2(t_next,a,b,c)
        next_point_yz = createPointinRobotSpace(next_point_yz[0],next_point_yz[1])
        next_point = (context_points_x[0],next_point_yz[0],next_point_yz[1], -1, 1, 0, 0, 1, 0)

        dist = distance((start_point[0], start_point[1], start_point[2]), (next_point[0], next_point[1], next_point[2]))
        time_needed = dist / speed
        now = time.time()
        end = now + time_needed

        while while_var and 0 < len(context_points_x) and cap.isOpened():
            state = bot.bridgeState
            eventList = state.split('"')
            #print(eventList)



            pose_dict = Int_exp01_cam.pose_dictionary
            #print (pose_dict)

            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)
            with torch.no_grad():
                input_image = torch.Tensor(input_image).cuda()

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                    heatmaps_result.squeeze(0),
                    offsets_result.squeeze(0),
                    displacement_fwd_result.squeeze(0),
                    displacement_bwd_result.squeeze(0),
                    output_stride=output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.15)
            keypoint_coords *= output_scale

            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)


            #cv2.imshow('posenet', overlay_image)
            coords = keypoint_coords[0]
            pose_dictionary['nose'] = coords[0]
            pose_dictionary['leftEye'] = coords[1]
            pose_dictionary['rightEye'] = coords[2]
            pose_dictionary['leftEar'] = coords[3]
            pose_dictionary['rightEar'] = coords[4]
            pose_dictionary['leftShoulder'] = coords[5]
            pose_dictionary['rightShoulder'] = coords[6]
            pose_dictionary['leftElbow'] = coords[7]
            pose_dictionary['rightElbow'] = coords[8]
            pose_dictionary['leftWrist'] = coords[9]
            pose_dictionary['rightWrist'] = coords[10]
            pose_dictionary['leftHip'] = coords[11]
            pose_dictionary['rightHip'] = coords[12]
            pose_dictionary['leftKnee'] = coords[13]
            pose_dictionary['rightKnee'] = coords[14]
            pose_dictionary['leftAnkle'] = coords[15]
            pose_dictionary['rightAnkle'] = coords[16]
            #print(pose_dictionary)

            ####   Experimenting with the rotation of sixth axis   ########################
            ### first exp: 0 and 5.0
            
            #list_point = list(next_point)
            #if (i % 2 == 0):
            #    list_point[4] = -10.0
            #else:
            #    list_point[4] = 10.0
            #next_point = tuple(list_point)
            

            # human scale
            rightEye = pose_dictionary['rightEye']
            leftEye = pose_dictionary['leftEye']
            human_scale = 6.4/(rightEye[1] - leftEye[1]) #multiply this to any pixel diff --> gives back centimeters

            # the whole camera integration goes here to add new points to the beginning of the context points list
            rightWrist = pose_dictionary['rightWrist']
            leftWrist = pose_dictionary['leftWrist']
            WristDiff = (rightWrist-leftWrist) * human_scale #centimeters- max 100 cm

            # calculate the angle between wrist-shoulder vector and vertical diraction
            rightShoulder = pose_dictionary['rightShoulder']
            vertical_dir = (0.0,1.0)
            rightWristVec = rightWrist - rightShoulder
            rightWristVec = normalizeVec(rightWristVec)
            dot_product = np.dot(rightWristVec, vertical_dir)
            rightWristBodyAngle = np.arccos(dot_product) #the result is between 0 and pi


            if (rightWrist[0]!=0 and rightWrist[1]!=0 and leftWrist[0]!=0 and leftWrist[1]!=0): #if the camera is seeing it basically

                # a can be xDiff - b can be yDiff and c can be the angle

                yDiff = WristDiff[0] * 0.3   # right and left difference - max 100
                xDiff = WristDiff[1] * 0.3   # up and down difference - max 100

                a = remap(xDiff,5.0,100.0,a_min,a_max)
                #print("a is", a)
                b = remap(yDiff,5.0,100.0,b_min,b_max)
                c = remap(rightWristBodyAngle, 0.0, math.pi, c_min, c_max)



            if time.time() >= end:  # meaning if the action is done
                bot.TransformTo(next_point[0], next_point[1], next_point[2], next_point[3], next_point[4],
                               next_point[5], next_point[6], next_point[7], next_point[8])


                current_point = next_point
                context_points_x.pop(0)

                t_next = remap(context_points_x[0], xmin, xmax, targetMin, targetMax)
                next_point_yz = hypocycloid2(t_next, a, b, c)
                next_point_yz = createPointinRobotSpace(next_point_yz[0], next_point_yz[1])
                next_point = (context_points_x[0], next_point_yz[0], next_point_yz[1], -1, 1, 0, 0, 1, 0)

                dist = distance((current_point[0], current_point[1], current_point[2]),
                                (next_point[0], next_point[1], next_point[2]))
                time_needed = dist / speed
                now = time.time()
                end = now + time_needed

    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit()



