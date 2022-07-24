import numpy as np
import turtle

leftArr = [0.706, 0.707, 0.706, 0.709, 0.71, 0.71, 0.71, 0.707, 0.707, 0.71, 0.711, 0.709, 0.711, 1.345, 1.355, 1.355,
           1.358, 1.358, 1.362, 1.354, 0.913, 0.642, 0.655, 0.678, 0.69, 0.697, 0.703, 0.706, 0.706, 0.71, 0.711, 0.708,
           0.713, 0.708, 0.711]
rightArr = [0.718, 0.712, 0.709, 0.71, 0.712, 0.709, 0.708, 0.71, 0.707, 0.71, 0.708, 0.711, 0.717, 0.644, 0.714, 0.64,
            0.717, 0.641, 0.715, 0.642, 0.716, 0.641, 0.718, 0.64, 0.718, 0.638, 0.718, 0.687, 0.685, 0.703, 0.714,
            0.719, 0.717, 0.716, 0.714, 0.713, 0.711, 0.71, 0.71, 0.709, 0.711, 0.385]


def getLeftSpeed(time):
    return 0.16 if 10 < time < 20 else 0.57


def getRightSpeed():
    return 0.57


def angle(left, right, distanse):
    if left == right:
        return 0
    return left / radius(left, right, distanse)[0]


def radius(left, right, distanse):
    return distanse / (right / left - 1), distanse / (right / left - 1) + distanse


distLeft = []
distRight = []
for i in range(min(len(rightArr), len(leftArr))):
    distLeft.append(leftArr[i] * getLeftSpeed(sum(leftArr[:1])))
    distRight.append(rightArr[i] * getRightSpeed())

distanse = 0.3
left = turtle.Turtle()
right = turtle.Turtle()
scale = 700
right.goto(distanse * scale * 0.1, 0)
left.left(90)
right.left(90)
for i in range(len(distRight)):
    if distLeft[i] == distRight[i]:
        left.forward(distLeft[i])
        right.forward(distLeft[i])
    else:
        left.circle(radius(distRight[i], distLeft[i], distanse)[0] * scale, np.abs(angle(distLeft[i], distRight[i], distanse)))
        right.circle(radius(distRight[i], distLeft[i], distanse)[0] * scale, np.abs(angle(distLeft[i], distRight[i], distanse)))
turtle.exitonclick()
