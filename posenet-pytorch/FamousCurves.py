from math import sin
from math import cos
from math import pi


### the way I imagin this is theat we have certain x values that
### the robot draws and the user changes the y and z based on these definitions

# functions return a tuple of y and z

def createT(minPI,maxPI,n):
    min = pi*minPI
    max = pi*maxPI
    step = (max-min)/n
    t = []
    for i in range(n):
        t.append(i*step)
    return t

# a and b are the scale in y and z - input t comes from createT
def lissajous(t,a,b,c,d,n):
    y = a * sin(n * t + c)
    z = b * sin(d * t)
    return y,z

def involute_circle(t,a,n):
    y = a * (cos(n * t) + t * sin(n * t))
    z = a * (sin(n * t) - t * cos(n * t))
    return y,z

def hypotrochoid(t,a,b,c):
    y = (a - b) * cos(t) + c * cos((a / b - 1) * t)
    z = (a - b) * sin(t) - c * sin((a / b - 1) * t)
    return y, z

def rose(t,a,m):
    y = a * cos(m * t) * cos(t)
    z = a * cos(m * t) * sin(t)
    return y, z

def archimedes_spiral(t,a,m):
    y = a * (t) * cos(m * t)
    z = a * (t) * sin(m * t)
    return y, z

def talbot(t,a,b,m):
    y = (a ** 2 + m ** 2 * sin(t) * sin(t)) * cos(t) / a
    z = (a ** 2 - 2 * m ** 2 + m ** 2 * sin(t) * sin(t)) * sin(t) / b
    return y, z

def epicycloid(t,a,b):
    y = (a + b) * cos(t) - b * cos((a / b + 1) * t)
    z = (a + b) * sin(t) - b * sin((a / b + 1) * t)
    return y, z

def epitrochoid(t,a,b,c):
    y = (a + b) * cos(t) - c * cos((a / b + 1) * t)
    z = (a + b) * sin(t) - c * sin((a / b + 1) * t)
    return y, z

def hypocycloid(t,a,b):
    y = (a - b) * cos(t) + b * cos((a / b - 1) * t)
    z = (a - b) * sin(t) - b * sin((a / b - 1) * t)
    return y, z


def hypocycloid2(t,a,b,c):
    y = (a - b) * cos(t) + c * cos((a / b - 1) * t)
    z = (a - b) * sin(t) - c * sin((a / b - 1) * t)
    return y, z