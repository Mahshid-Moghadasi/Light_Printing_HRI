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


def start_pose_right_gersture():
    if -20<radiansTodegrees(twoPointsAngle(rightWrist,rightElbow))<20 and 75<radiansTodegrees(twoPointsAngle(rightElbow, rightShoulder))<105:
        return True
    return False

def start_pose_left_gesture():
    if -20<radiansTodegrees(twoPointsAngle(leftWrist,leftElbow))<20 and 75<radiansTodegrees(twoPointsAngle(leftElbow, leftShoulder))<105:
        return True
    return False

def done_gesture():
    if start_pose_left_gesture() and start_pose_right_gersture():
        return True
    return False
