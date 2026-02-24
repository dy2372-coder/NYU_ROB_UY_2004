# my_assignment_2.py
import numpy as np

# Link distances (meters)
D12 = 0.3
D23 = 0.4
D34 = 0.3

# ----------------------------
# Rotation helpers (RHR)
# ----------------------------
def rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)

def roty(theta: float) -> np.ndarray:
    """
    Rotation about y that matches the provided unit tests:
        [[cos, 0, -sin],
         [  0, 1,   0 ],
         [sin, 0,  cos]]
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c,  0.0, -s],
        [0.0, 1.0, 0.0],
        [s,  0.0,  c]
    ], dtype=float)

def T_from_R_p(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = R
    T[0:3, 3] = p.reshape(3,)
    return T

# ----------------------------
# Required transforms
# ----------------------------
def get_T01(theta_1: float) -> np.ndarray:
    # Frame 1 rotates relative to frame 0 by theta_1 about z0, no translation
    return T_from_R_p(rotz(theta_1), np.array([0.0, 0.0, 0.0]))

def get_T12(theta_2: float) -> np.ndarray:
    # Frame 2 rotates relative to frame 1 by theta_2 about y1
    # Origin offset from 1 to 2 is +0.3 along y1
    return T_from_R_p(roty(theta_2), np.array([0.0, D12, 0.0]))

def get_T23(theta_3: float) -> np.ndarray:
    # Frame 3 rotates relative to frame 2 by theta_3 about y2
    # Origin offset from 2 to 3 is +0.4 along x2
    return T_from_R_p(roty(theta_3), np.array([D23, 0.0, 0.0]))

def get_T34() -> np.ndarray:
    # Fixed transform from 3 to 4: +0.3 along z3, no rotation
    return T_from_R_p(np.eye(3), np.array([0.0, 0.0, D34]))

def get_FK(theta_1: float, theta_2: float, theta_3: float) -> np.ndarray:
    # Homogeneous transform from frame 0 to frame 4
    return get_T01(theta_1) @ get_T12(theta_2) @ get_T23(theta_3) @ get_T34()

# ----------------------------
# Collision functions
# ----------------------------
def ee_in_collision(theta_list, p_point: np.ndarray, tolerance: float) -> bool:
    """
    Returns True if distance between EE position and p_point is < tolerance.
    Inputs:
      - theta_list: [theta_1, theta_2, theta_3]
      - p_point: 3x1 or (3,) numpy array (position in frame 0)
      - tolerance: scalar (meters)
    """
    theta_1, theta_2, theta_3 = theta_list
    FK = get_FK(theta_1, theta_2, theta_3)
    p_ee = FK[0:3, 3]  # (3,)
    p_point = np.array(p_point, dtype=float).reshape(3,)
    dist = np.linalg.norm(p_ee - p_point)
    return dist < tolerance

def path_in_collision(path, object_list, tolerance: float = 0.0) -> bool:
    """
    Returns True if for ANY configuration in 'path', the EE is inside
    (radius + tolerance) of ANY spherical object in object_list.

    Inputs:
      - path: list of tuples [(theta1,theta2,theta3), ...]
      - object_list: list of tuples [(center, radius), ...]
          where center is 3x1 or (3,) numpy array in frame 0
      - tolerance: scalar (meters)

    Interpretation (standard/sensible):
      collision if ||p_ee - center|| < (radius + tolerance)
    """
    for (t1, t2, t3) in path:
        FK = get_FK(t1, t2, t3)
        p_ee = FK[0:3, 3].reshape(3,)
        for (center, radius) in object_list:
            center = np.array(center, dtype=float).reshape(3,)
            if np.linalg.norm(p_ee - center) < (float(radius) + float(tolerance)):
                return True
    return False



