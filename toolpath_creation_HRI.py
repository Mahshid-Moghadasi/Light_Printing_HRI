from machinaRobot import *
import sys
import torch
import cv2
import time
import argparse
import posenet
import Int_exp01_cam
import math
import numpy as np
from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM, SOCK_STREAM

###################################### MATH FUNCTIONS ###############################################################

def distance3D(pointA, pointB):
    d = (pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2 + (pointA[2] - pointB[2]) ** 2
    d = math.sqrt(d)
    return d


def distance2D(pointA, pointB):
    d = (pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2
    d = math.sqrt(d)
    return d


def remap(x, xmin, xmax, targetMin, targetMax):
    if x <= xmin: return targetMin
    if x >= xmax: return targetMax
    return (x - xmin) / (xmax - xmin) * (targetMax - targetMin) + targetMin


def createPointinRobotSpace(y, z):
    y = y * 20
    z = z * 20
    z = z + 400
    return y, z


def normalizeVec(vec):
    return vec / np.linalg.norm(vec)


def switchBool(x):
    if x == True:
        return False
    return True


# calculates the angle between two-point vector and vertical diraction - returns a number between 0 and pi
def twoPointsAngle(point1, point2):
    vector = point2 - point1
    vector = normalizeVec(vector)
    dot_product = np.dot(vector, vertical_dir)
    return np.arccos(dot_product)  # the result is between 0 and pi


def radiansTodegrees(x):
    return x * 180 / math.pi

###########################################  ROBOT CHECK FUNCTIONS   ################################################################

def checkX(x):
    if x < 200:
        return 200
    elif x > 500:
        return 500
    else:
        return x


def checkY(y):
    if y < -450:
        return -450
    elif y > 450:
        return 450
    else:
        return y


def checkZ(z):
    if z < 160:
        return 160
    elif z > 550:
        return 550
    else:
        return z
############################################# HUMAN_ROBOT INTERACTION FUNCTIONS #################################################################

def square_gesture():
    if 0 < radiansTodegrees(twoPointsAngle(leftWrist, leftElbow)) < 25 and 80 < radiansTodegrees(
            twoPointsAngle(leftElbow, leftShoulder)) < 120 and 0 < radiansTodegrees(twoPointsAngle(rightWrist, rightElbow)) < 25 and 80 < radiansTodegrees(
            twoPointsAngle(rightElbow, rightShoulder)) < 120:
        return True
    return False

def triangle_gesture():
    if 25< radiansTodegrees(twoPointsAngle(leftWrist, leftElbow)) < 55 and 135 < radiansTodegrees(
            twoPointsAngle(leftElbow, leftShoulder)) < 165 and 25 < radiansTodegrees(twoPointsAngle(rightWrist, rightElbow)) < 55 and 135 < radiansTodegrees(
            twoPointsAngle(rightElbow, rightShoulder)) < 165:
        return True
    return False

def circle_gesture():
    if 0< radiansTodegrees(twoPointsAngle(leftWrist, leftElbow)) < 50 and 30 < radiansTodegrees(
            twoPointsAngle(leftElbow, leftShoulder)) < 65 and 0 < radiansTodegrees(twoPointsAngle(rightWrist, rightElbow)) < 50 and 30 < radiansTodegrees(
            twoPointsAngle(rightElbow, rightShoulder)) < 65:
        return True
    return False

def draw(list, scale):
    for i in range(0,len(list)):
        point = list[i]
        bot.MoveTo(point[0]*scale,point[1]*scale,point[2]*scale)

def gesture():
    if square_gesture():
        return 'square'
    elif triangle_gesture():
        return 'triangle'
    elif circle_gesture():
        return 'circle'
    else:
        return 'nothing'

def all_joints_visible():
    if rightWrist[0] != 0 and rightWrist[1] != 0 and leftWrist[0] != 0 and leftWrist[1] != 0 and leftShoulder[1] != 0 and \
            rightShoulder[1] != 0 and leftShoulder[0] != 0 and rightShoulder[0] != 0 and rightElbow[0]!=0 and rightElbow[1]!=0:
        return True
    return False

############################################# CAMERA SETTINGS #################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)  # default=0.7125)
args = parser.parse_args()

pose_dictionary = {'nose': [0, 0],
                   'leftEye': [0, 0],
                   'rightEye': [0, 0],
                   'leftEar': [0, 0],
                   'rightEar': [0, 0],
                   'leftShoulder': [0, 0],
                   'rightShoulder': [0, 0],
                   'leftElbow': [0, 0],
                   'rightElbow': [0, 0],
                   'leftWrist': [0, 0],
                   'rightWrist': [0, 0],
                   'leftHip': [0, 0],
                   'rightHip': [0, 0],
                   'leftKnee': [0, 0],
                   'rightKnee': [0, 0],
                   'leftAnkle': [0, 0],
                   'rightAnkle': [0, 0]
                   }
########################################### ROBOT SETTINGS ###################################################################
bot = MachinaRobot()
speed = 30

########################################### INTERACTION SETTINGS  ############################################################
# Interaction settings
vertical_dir = (1.0, 0.0)
states = []  #recording maybe 3 consecutive states
state1 = 'nothing'
state2 = 'nothing'
stop = False

circle = [(0,-0.0192147178166427,-0.195090313066826),(0,-0.0569057483793858,-0.187593116177447),(0,-0.0924098964384434,-0.172886766265923),(0,-0.124362850051947,-0.151536579549318),(0,-0.151536555819826,-0.12436283826257),(0,-0.17288684022947,-0.0924099362137129),(0,-0.187593072702387,-0.0569057315545229),(0,-0.195090318561898,-0.0192147189096805),(0,-0.195090313066827,0.0192147178166431),(0,-0.187593116177446,0.056905748379386),(0,-0.172886766265917,0.0924098964384398),(0,-0.151536579549322,0.12436285005195),(0,-0.124362838262571,0.151536555819827),(0,-0.0924099362137124,0.17288684022947),(0,-0.0569057315545232,0.187593072702388),(0,-0.0192147189096802,0.195090318561897),(0,0.0192147178166427,0.195090313066826),(0,0.0569057483793863,0.187593116177448),(0,0.0924098964384352,0.17288676626591),(0,0.124362850051954,0.151536579549329),(0,0.151536555819827,0.124362838262571),(0,0.17288684022947,0.0924099362137124),(0,0.187593072702388,0.0569057315545232),(0,0.195090318561897,0.0192147189096802),(0,0.195090313066826,-0.0192147178166426),(0,0.187593116177448,-0.0569057483793863),(0,0.172886766265923,-0.0924098964384434),(0,0.151536579549317,-0.124362850051946),(0,0.124362838262571,-0.151536555819828),(0,0.0924099362137123,-0.17288684022947),(0,0.0569057315545231,-0.187593072702388),(0,0.0192147189096802,-0.195090318561897)]
triangle = [(0,-1.12583302491977,4.44089209850063E-16),(0,-1.12583302491977,4.44089209850063E-16),(0,0.562916512459885,0.975),(0,0.562916512459885,0.975),(0,0.562916512459885,-0.975000000000001),(0,0.562916512459884,-0.975)]
square = [(0,-1,0),(0,-1,0),(0,0,1),(0,0,1),(0,1,0),(0,1,0),(0,0,-1),(0,0,-1)]
drawing_list = []
scale = 50.0
start_point = (350, 100 ,400, -1, 1, 0, 0, 1, 0)

########################################### START COMMANDS  #################################################################
#### Run these before code start
# bot.AxesTo(0, 0, 0, 0, 90, 45)
# bot.TransformTo(start_point)

###############################################   MAIN    ##################################################################
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
        next_move = (0,0,0)
        next_point = (start_point[0]+next_move[0], start_point[1]+next_move[1], start_point[2]+next_move[2])
        current_point = next_point
        dist = distance2D((start_point[0], start_point[1], start_point[2]), (next_point[0], next_point[1], next_point[2]))
        time_needed = dist / speed
        now = time.time()
        end = now + time_needed


        while not stop and cap.isOpened():
            pose_dict = Int_exp01_cam.pose_dictionary
            # print (pose_dict)

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
                    max_pose_detections=1,
                    min_pose_score=0.1)
            keypoint_coords *= output_scale

            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.1, min_part_score=0.1)

            # cv2.imshow('posenet', overlay_image)
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
            # print(pose_dictionary)


            # the whole camera integration goes here to add new points to the beginning of the context points list
            global rightWrist
            global leftWrist
            global rightShoulder
            global leftShoulder
            global rightElbow
            global leftElbow
            rightWrist = pose_dictionary['rightWrist']
            leftWrist = pose_dictionary['leftWrist']
            rightShoulder = pose_dictionary['rightShoulder']
            leftShoulder = pose_dictionary['leftShoulder']
            rightElbow = pose_dictionary['rightElbow']
            leftElbow = pose_dictionary['leftElbow']

            time.sleep(0.2)
            #print(radiansTodegrees(twoPointsAngle(leftWrist, leftElbow)))
            #rint(radiansTodegrees(twoPointsAngle(leftElbow, leftShoulder)))
            #print(radiansTodegrees(twoPointsAngle(rightWrist, rightElbow)))
            #print(radiansTodegrees(twoPointsAngle(rightElbow, rightShoulder)))
            #print(gesture())

            #if all_joints_visible(): #if the camera is seeing it basically
            state1 = gesture()
            print("state 1 is: ", state1, "state 2 is: ", state2)
            """
            if len(states)==10:
                if state[0] == state[1] == state[2] == state[3]== state[4]== state[5]== state[6]== state[7]== state[8]== state[9]:
                    final_state = state[9]
                states.pop(0)
            states.append(state)
            print(final_state)
            """

            if state1 != 'nothing' and state1 != state2:
                if state1 == 'circle':
                    drawing_list = circle.copy()
                elif state1 == 'triangle':
                    drawing_list = triangle.copy()
                elif state1 == 'square':
                    drawing_list = square.copy()

                next_move = drawing_list[0]
                next_point = (current_point[0] + next_move[0]*scale, current_point[1] + next_move[1]*scale, current_point[2] + next_move[2]*scale)


            if time.time() >= end:  # meaning if the action is done
                bot.MoveTo(next_point[0], next_point[1], next_point[2])
                current_point = next_point

                if len(drawing_list)>1:
                    drawing_list.pop(0)
                    next_move = drawing_list[0]
                    next_point = (current_point[0] + next_move[0]*scale, current_point[1] + next_move[1]*scale, current_point[2] + next_move[2]*scale)

                    dist = distance2D((current_point[0], current_point[1], current_point[2]),
                                (next_point[0], next_point[1], next_point[2]))
                    time_needed = dist / speed
                    now = time.time()
                    end = now + time_needed

            state2 = state1


    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit()
