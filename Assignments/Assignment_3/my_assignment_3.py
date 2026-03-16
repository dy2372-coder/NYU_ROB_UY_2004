import math
import numpy as np
import scipy


def forward_kinematics(theta1, theta2, theta3):
    def rotation_x(angle):
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ])

    def rotation_y(angle):
        return np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1]
        ])

    def rotation_z(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def translation(x, y, z):
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

    T_0_1 = translation(0.07500, -0.08350, 0) @ rotation_x(1.57080) @ rotation_z(theta1)
    T_1_2 = rotation_y(-1.57080) @ rotation_z(theta2)
    T_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta3)
    T_3_ee = translation(0.06231, -0.06216, 0.01800)
    T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
    return T_0_ee[:3, 3]


def get_cost(theta, target_position):
    theta = np.array(theta, dtype=float).reshape(3,)
    target_position = np.array(target_position, dtype=float).reshape(3,)

    ee_position = forward_kinematics(theta[0], theta[1], theta[2])
    error = target_position - ee_position

    C = np.sum(error ** 2)
    mean_abs_error = np.mean(np.abs(error))

    return C, mean_abs_error


def get_gradient(theta, target_position):
    theta = np.array(theta, dtype=float).reshape(3,)
    target_position = np.array(target_position, dtype=float).reshape(3,)

    epsilon = 1e-6
    gradient = np.zeros(3)

    for i in range(3):
        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon

        C_plus, _ = get_cost(theta_plus, target_position)
        C_minus, _ = get_cost(theta_minus, target_position)

        gradient[i] = (C_plus - C_minus) / (2 * epsilon)

    return gradient


def inverse_kinematics_with_optimizer(target_position):
    target_position = np.array(target_position, dtype=float).reshape(3,)

    def objective(theta):
        C, _ = get_cost(theta, target_position)
        return C

    theta0 = np.zeros(3)
    result = scipy.optimize.minimize(objective, theta0, method='BFGS')

    return result.x


def inverse_kinematics_with_gradient(target_position):
    target_position = np.array(target_position, dtype=float).reshape(3,)

    theta = np.zeros(3)
    learning_rate = 0.01
    max_steps = 5000
    tolerance = 1e-5

    for _ in range(max_steps):
        _, mean_error = get_cost(theta, target_position)

        if mean_error < tolerance:
            break

        gradient = get_gradient(theta, target_position)
        theta = theta - learning_rate * gradient

    result = scipy.optimize.minimize(
        lambda x: get_cost(x, target_position)[0],
        theta,
        method='BFGS'
    )

    return result.x