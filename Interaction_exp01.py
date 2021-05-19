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



def main():
    bot.Message("Hello Robot!")
    bot.SpeedTo(100)
    bot.AxesTo(0, 0, 0, 0, 90, 0)
    bot.TransformTo(200, 300, 200, -1, 0, 0, 0, 1, 0)
    bot.Rotate(0, 1, 0, -90)
    bot.Move(0, 0, 250)
    bot.Wait(2000)
    bot.AxesTo(0, 0, 0, 0, 90, 0)
    # global actions
    # actions.append("first action")


def distance(pointA, pointB):
    d = (pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2 + (pointA[2] - pointB[2]) ** 2
    d = math.sqrt(d)
    return d

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
start_point = (366.07, 147.61, 223.36, -1, 1, 0, 0, 1, 0)  # rotating the sixth axis: (300, 0, 300, -1, this one, 0, 0, 1, 0) It's good practice to go between -10 and 10
context_points = [(366.07,147.61,223.36, -1, 1, 0, 0, 1, 0),(366.07,120.46,223.36, -1, 1, 0, 0, 1, 0),
                  (366.07,93.31,223.36, -1, 1, 0, 0, 1, 0),(366.07,66.16,223.36, -1, 1, 0, 0, 1, 0),
                  (366.07,39.01,223.36, -1, 1, 0, 0, 1, 0),(366.07,11.86,223.36, -1, 1, 0, 0, 1, 0),
                  (366.07,-15.3,223.36, -1, 1, 0, 0, 1, 0),(366.07,-42.45,223.36, -1, 1, 0, 0, 1, 0),
                  (366.07,-69.6,223.36, -1, 1, 0, 0, 1, 0),(366.07,-96.75,223.36, -1, 1, 0, 0, 1, 0),
                  (366.07,-123.9,223.36, -1, 1, 0, 0, 1, 0),(366.07,-123.9,274.92, -1, 1, 0, 0, 1, 0),
                  (366.07,-96.75,274.92, -1, 1, 0, 0, 1, 0),(366.07,-69.6,274.92, -1, 1, 0, 0, 1, 0),
                  (366.07,-42.45,274.92, -1, 1, 0, 0, 1, 0),(366.07,-15.3,274.92, -1, 1, 0, 0, 1, 0),
                  (366.07,11.86,274.92, -1, 1, 0, 0, 1, 0),(366.07,39.01,274.92, -1, 1, 0, 0, 1, 0),
                  (366.07,66.16,274.92, -1, 1, 0, 0, 1, 0),(366.07,93.31,274.92, -1, 1, 0, 0, 1, 0),
                  (366.07,120.46,274.92, -1, 1, 0, 0, 1, 0),(366.07,147.61,274.92, -1, 1, 0, 0, 1, 0),
                  (366.07,147.61,326.49, -1, 1, 0, 0, 1, 0),(366.07,120.46,326.49, -1, 1, 0, 0, 1, 0),
                  (366.07,93.31,326.49, -1, 1, 0, 0, 1, 0),(366.07,66.16,326.49, -1, 1, 0, 0, 1, 0),
                  (366.07,39.01,326.49, -1, 1, 0, 0, 1, 0),(366.07,11.86,326.49, -1, 1, 0, 0, 1, 0),
                  (366.07,-15.3,326.49, -1, 1, 0, 0, 1, 0),(366.07,-42.45,326.49, -1, 1, 0, 0, 1, 0),
                  (366.07,-69.6,326.49, -1, 1, 0, 0, 1, 0),(366.07,-96.75,326.49, -1, 1, 0, 0, 1, 0),
                  (366.07,-123.9,326.49, -1, 1, 0, 0, 1, 0),(366.07,-123.9,378.06, -1, 1, 0, 0, 1, 0),
                  (366.07,-96.75,378.06, -1, 1, 0, 0, 1, 0),(366.07,-69.6,378.06, -1, 1, 0, 0, 1, 0),
                  (366.07,-42.45,378.06, -1, 1, 0, 0, 1, 0),(366.07,-15.3,378.06, -1, 1, 0, 0, 1, 0),
                  (366.07,11.86,378.06, -1, 1, 0, 0, 1, 0),(366.07,39.01,378.06, -1, 1, 0, 0, 1, 0),
                  (366.07,66.16,378.06, -1, 1, 0, 0, 1, 0),(366.07,93.31,378.06, -1, 1, 0, 0, 1, 0),
                  (366.07,120.46,378.06, -1, 1, 0, 0, 1, 0),(366.07,147.61,378.06, -1, 1, 0, 0, 1, 0),
                  (366.07,147.61,429.62, -1, 1, 0, 0, 1, 0),(366.07,120.46,429.62, -1, 1, 0, 0, 1, 0),
                  (366.07,93.31,429.62, -1, 1, 0, 0, 1, 0),(366.07,66.16,429.62, -1, 1, 0, 0, 1, 0),
                  (366.07,39.01,429.62, -1, 1, 0, 0, 1, 0),(366.07,11.86,429.62, -1, 1, 0, 0, 1, 0),
                  (366.07,-15.3,429.62, -1, 1, 0, 0, 1, 0),(366.07,-42.45,429.62, -1, 1, 0, 0, 1, 0),
                  (366.07,-69.6,429.62, -1, 1, 0, 0, 1, 0),(366.07,-96.75,429.62, -1, 1, 0, 0, 1, 0),
                  (366.07,-123.9,429.62, -1, 1, 0, 0, 1, 0),(366.07,-123.9,481.19, -1, 1, 0, 0, 1, 0),
                  (366.07,-96.75,481.19, -1, 1, 0, 0, 1, 0),(366.07,-69.6,481.19, -1, 1, 0, 0, 1, 0),
                  (366.07,-42.45,481.19, -1, 1, 0, 0, 1, 0),(366.07,-15.3,481.19, -1, 1, 0, 0, 1, 0),
                  (366.07,11.86,481.19, -1, 1, 0, 0, 1, 0),(366.07,39.01,481.19, -1, 1, 0, 0, 1, 0),
                  (366.07,66.16,481.19, -1, 1, 0, 0, 1, 0),(366.07,93.31,481.19, -1, 1, 0, 0, 1, 0),
                  (366.07,120.46,481.19, -1, 1, 0, 0, 1, 0),(366.07,147.61,481.19, -1, 1, 0, 0, 1, 0)]


speed = 100
bot = MachinaRobot()
start = time.time()
while_var = True

global action
prev_action = "nothing to do"
action = "nothing to do"


############################################################################################################################
# Interaction settings
# y = a* sin(x/b)
max_a = 50
max_b = 50

########  run these at the start   ###########
# bot.AxesTo(0, 0, 0, 0, 90, 0)
# bot.TransformTo(start_point)

async def feedback():
    address = "ws://127.0.0.1:6999/Bridge"
    async with websockets.connect(address) as websocket:
        f = await websocket.recv()
        return f


if __name__ == '__main__':
    try:

        # state = feedback()
        # print(state)

        # camera settings
        cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        model = posenet.load_model(args.model)
        model = model.cuda()
        output_stride = model.output_stride


        # starting move
        bot.SpeedTo(speed)
        next_point = context_points[0]
        dist = distance((start_point[0], start_point[1], start_point[2]), (next_point[0], next_point[1], next_point[2]))
        time_needed = dist / speed
        now = time.time()
        end = now + time_needed


        while while_var and 0 < len(context_points) and cap.isOpened():
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
            pose_dictionary['rightEye'] = coords[1]
            pose_dictionary['leftEye'] = coords[2]
            pose_dictionary['rightEar'] = coords[3]
            pose_dictionary['leftEar'] = coords[4]
            pose_dictionary['rightShoulder'] = coords[5]
            pose_dictionary['leftShoulder'] = coords[6]
            pose_dictionary['rightElbow'] = coords[7]
            pose_dictionary['leftElbow'] = coords[8]
            pose_dictionary['rightWrist'] = coords[9]
            pose_dictionary['leftWrist'] = coords[10]
            pose_dictionary['rightHip'] = coords[11]
            pose_dictionary['leftHip'] = coords[12]
            pose_dictionary['rightKnee'] = coords[13]
            pose_dictionary['leftKnee'] = coords[14]
            pose_dictionary['rightAnkle'] = coords[15]
            pose_dictionary['leftAnkle'] = coords[16]
            #print(pose_dictionary)

            ####   Experimenting with the rotation of sixth axis   ########################
            ### first exp: 0 and 5.0
            """
            list_point = list(next_point)
            if (i % 2 == 0):
                list_point[4] = -10.0
            else:
                list_point[4] = 10.0
            next_point = tuple(list_point)
            """

            # human scale
            rightEye = pose_dictionary['rightEye']
            leftEye = pose_dictionary['leftEye']
            human_scale = 6.4/(rightEye[1] - leftEye[1]) #multiply this to any pixel diff --> gives back centimeters


            # the whole camera integration goes here to add new points to the beginning of the context points list
            rightWrist = pose_dictionary['rightWrist']
            leftWrist = pose_dictionary['leftWrist']
            WristDiff = (rightWrist-leftWrist) * human_scale #centimeters- max 100 cm


            if (rightWrist[0]!=0 and rightWrist[1]!=0 and leftWrist[0]!=0 and leftWrist[1]!=0): #if the camera is seeing it basically

                #sin_a = ((wristDistanceX-200) / (500-200)) * 3
                #sin_b = ((wristDistanceY-160) / (680-160)) * 3

                y = WristDiff[0] * 0.3   # right and left difference - max 100
                x = WristDiff[1] * 0.3   # up and down difference - max 100

                list_point = list(next_point)
                list_point[0] += y
                list_point[2] += x

                # checking if it's within range
                list_point[0] = checkX(list_point[0])
                list_point[2] = checkZ(list_point[2])

                next_point = tuple(list_point)
                #print('sin-a', sin_a)
                #print('sin_b', sin_b)


                #xDiff = current_point[0] - prev_point[0]
                #yDiff = current_point[1] - prev_point[1]
                #DiffStep = yDiff/25
                #Diff = current_point[2] - prev_point[2]

                # at this stage we just make change to z and create 10 intermediate points between y0 and y1
                #yDiff = prev_point[1] + yDiffStep
                #for i in range(0,5):
                    #new_point = (current_point[0],current_point[1]+i*yDiffStep,sin_a+current_point[2], -1, 1, 0, 0, 1, 0)
                    #context_points.insert(i,new_point)

            if time.time() >= end:  # meaning if the action is done
                bot.TransformTo(next_point[0], next_point[1], next_point[2], next_point[3], next_point[4],
                                next_point[5], next_point[6], next_point[7], next_point[8])

                current_point = next_point
                context_points.pop(0)


                next_point = context_points[0]
                dist = distance((current_point[0], current_point[1], current_point[2]),
                                (next_point[0], next_point[1], next_point[2]))
                time_needed = dist / speed
                now = time.time()
                end = now + time_needed


    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit()



