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
from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM , SOCK_STREAM

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
    if x==True:
        return False
    return True

# calculates the angle between two-point vector and vertical diraction - returns a number between 0 and pi
def twoPointsAngle(point1,point2):
    vector = point2-point1
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

def start_pose_right_gersture():
    if -20<radiansTodegrees(twoPointsAngle(rightWrist,rightElbow))<20 and 75<radiansTodegrees(twoPointsAngle(rightElbow, rightShoulder))<105:
        return True
    return False

def start_pose_left_gesture():
    if -20<radiansTodegrees(twoPointsAngle(leftWrist,leftElbow))<20 and 75<radiansTodegrees(twoPointsAngle(leftElbow, leftShoulder))<105:
        return True
    return False

def square_gesture():
    if start_pose_left_gesture() and start_pose_right_gersture():
        return True
    return False

def square_draw():
    bot.SpeedTo(150)

def circle_draw():
    bot.SpeedTo(200)
    
def triangle_draw():
    bot.SpeedTo(150)

#############################################  UDP SETTINGS  #################################################################
port = 5000
SIZE = 128

DomIP = '10.48.56.122' # Laptop 1 IP - Dom
SubIP = '10.48.8.252' # Laptop 2 IP - Sub

def UDP_send(IP,port,message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #IPv4 DNS server - UDP protocol
    sock.sendto(bytes(message, "utf-8"), (IP, port))#self, data, address

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
start = time.time()

########################################### INTERACTION SETTINGS  ############################################################
# Interaction settings
context_points_right = []
context_points_left = []
vertical_dir = (1.0, 0.0)

path = r'C:\Users\mogha\PycharmProjects\Camera_Calibration\posenet-pytorch\toolpath.yaml'
cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)





cv_file.release()

collecting_points_right = False
collecting_points_left = False
right_stop = False
left_stop = False
stop = False

########################################### START COMMANDS  #################################################################
#### Run these before code start
# bot.AxesTo(0, 0, 0, 0, 90, 0)
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
            # print(pose_dictionary)

            # human scale
            rightEye = pose_dictionary['rightEye']
            leftEye = pose_dictionary['leftEye']
            human_scale = 6.4 / (rightEye[1] - leftEye[1])  # multiply this to any pixel diff --> gives back centimeters

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

            #print("start pose right:   ", start_pose_right())
            #print("start pose left:   ", start_pose_left())
            #print("right Elbow:   ", rightElbow[0])
            #time.sleep(0.1)
            #print("right Wrist:   ", rightWrist)
            #time.sleep(1)


    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit()



