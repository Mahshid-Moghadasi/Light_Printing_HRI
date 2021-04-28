import torch
import cv2
import time
import argparse
import posenet

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

def pose():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    start = time.time()
    frame_count = 0
    while True:
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

        cv2.imshow('posenet', overlay_image)
        frame_count += 1

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

        # human scale
        rightEye = pose_dictionary['rightEye']
        leftEye = pose_dictionary['leftEye']
        human_scale = 6.4 / (rightEye[1] - leftEye[1])  # multiply this to any pixel diff --> gives back centimeters

        # the whole camera integration goes here to add new points to the beginning of the context points list
        rightWrist = pose_dictionary['rightWrist']
        leftWrist = pose_dictionary['leftWrist']


        print("rightWrist", rightWrist*human_scale, "leftWrist", leftWrist*human_scale)
        print("dif", (rightWrist-leftWrist)*human_scale)
        #print("wristDistanceY", wristDistanceY)

        if (rightWrist[0] != 0 and rightWrist[1] != 0 and leftWrist[0] != 0 and leftWrist[1] != 0):
            wristDistanceX = rightWrist[1] - leftWrist[1]
            wristDistanceY = rightWrist[0] - leftWrist[0]

            sin_a = ((wristDistanceX - 200) / (500 - 200)) * 3
            sin_b = ((wristDistanceY - 160) / (680 - 160)) * 3
        #print (pose_dictionary['nose'])

        #print("wristDistanceX", wristDistanceX)
        #print("wristDistanceY", wristDistanceY)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == "__main__":
    pose()


