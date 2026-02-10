import numpy as np
import math

def rotate2D(theta, p_point):
    p = np.array(p_point, dtype=float)
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    return np.matmul(R, p)

def rotate3D(theta, axis_of_rotation, p_point):
    p = np.array(p_point, dtype=float)
    c = math.cos(theta)
    s = math.sin(theta)
    axis = axis_of_rotation.lower()

    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s,  c]], dtype=float)
    elif axis == 'y':
        R = np.array([[ c, 0,  s],
                      [ 0, 1,  0],
                      [-s, 0,  c]], dtype=float)
    elif axis == 'z':
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]], dtype=float)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    return np.matmul(R, p)

def rotate3D_many_times(rotation_list, p_point):
    q = np.array(p_point, dtype=float)
    for theta, axis in rotation_list:
        q = rotate3D(theta, axis, q)
    return q
