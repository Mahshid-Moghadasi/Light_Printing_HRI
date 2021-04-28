from machinaRobot import *
import sys
import time
import re
import math


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


context_points = [(200, 250, 200, -1, 0, 0, 0, 1, 0), (200, 500, 200, -1, 0, 0, 0, 1, 0), (300, 300, 300, -1, 0, 0, 0, 1, 0), (200, 300, 400, -1, 0, 0, 0, 1, 0)]
executed = []

bot = MachinaRobot()
global action
prev_action = "nothing to do"
action = "nothing to do"

if __name__ == '__main__':
    print("1", bot.bridgeState)
    try:

        while True:
            #main()
            event = bot.bridgeState
            eventList = event.split('"')
            print(eventList)



    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit()

