from machinaRobot import *
import sys
import re
import math
import time
import websockets
import asyncio



def main():
    bot.Message("Hello Robot!")
    bot.SpeedTo(100)
    bot.AxesTo(0,0,0,0,90,0)
    bot.TransformTo(200, 300, 200, -1, 0, 0, 0, 1, 0)
    bot.Rotate(0,1,0,-90)
    bot.Move(0,0,250)
    bot.Wait(2000)
    bot.AxesTo(0,0,0,0,90,0)
    #global actions
    #actions.append("first action")


def distance(pointA,pointB):
    d = (pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2 + (pointA[2] - pointB[2])**2
    d = math.sqrt(d)
    return d


start_point = (300, 0, 300, -1, 0, 0, 0, 1, 0) #rotating the sixth axis: (300, 0, 300, -1, this one, 0, 0, 1, 0) It's good practice to go between -10 and 10
context_points = [(366.07,147.61,223.36, -1, 1, 0, 0, 1, 0),(366.07,-123.9,223.36, -1, 1, 0, 0, 1, 0),(366.07,-123.9,274.92, -1, 1, 0, 0, 1, 0),(366.07,147.61,274.92, -1, 1, 0, 0, 1, 0),(366.07,147.61,326.49, -1, 1, 0, 0, 1, 0),(366.07,-123.9,326.49, -1, 1, 0, 0, 1, 0),(366.07,-123.9,378.06, -1, 1, 0, 0, 1, 0),(366.07,147.61,378.06, -1, 1, 0, 0, 1, 0),(366.07,147.61,429.62, -1, 1, 0, 0, 1, 0),(366.07,-123.9,429.62, -1, 1, 0, 0, 1, 0),(366.07,-123.9,481.19, -1, 1, 0, 0, 1, 0),(366.07,147.61,481.19, -1, 1, 0, 0, 1, 0)]

current_action = []
speed = 100

bot = MachinaRobot()
start = time.time()
while_var = True

global action
prev_action = "nothing to do"
action = "nothing to do"

########  run these at the start   ###########
#bot.AxesTo(0, 0, 0, 0, 90, 0)
#bot.TransformTo(start_point)

async def feedback():
    address = "ws://127.0.0.1:6999/Bridge"
    async with websockets.connect(address) as websocket:
        f = await websocket.recv()
        return f

if __name__ == '__main__':
    try:

        #state = feedback()
        #print(state)

        #starting move
        i = 0
        bot.SpeedTo(speed)


        next_point = context_points[i]

        dist = distance((start_point[0],start_point[1], start_point[2]), (next_point[0],next_point[1],next_point[2]))
        time_needed = dist / speed

        now = time.time()
        end = now + time_needed


        while while_var and i<len(context_points):
            state = bot.bridgeState
            eventList = state.split('"')
            print(eventList)

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


            if time.time() >= end: #meaning if the action is done
                bot.TransformTo(next_point[0], next_point[1], next_point[2], next_point[3], next_point[4],
                                next_point[5], next_point[6], next_point[7], next_point[8])

                current_point = next_point
                context_points.pop(0)

                #the whole camera integration goes here to add new points to the beginning of the context points list


                next_point = context_points[0]
                dist = distance((current_point[0],current_point[1], current_point[2]), (next_point[0],next_point[1],next_point[2]))
                time_needed = dist / speed
                now = time.time()
                end = now + time_needed

    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit()

