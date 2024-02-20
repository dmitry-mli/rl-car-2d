from rl.apps.car.common.types import Vector


def left(vector: Vector) -> Vector:
    x, y = vector
    return y, -x


def opposite(vector: Vector) -> Vector:
    x, y = vector
    return -x, -y


def right(vector: Vector) -> Vector:
    x, y = vector
    return -y, x
