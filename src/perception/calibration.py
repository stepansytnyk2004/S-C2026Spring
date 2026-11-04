import json

import cv2
import numpy as np


### sub-pixel refinement stopping rule, as in week 3
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


### checkerboard corners in one image, n x 2 sub-pixel, or none if not found
def find_corners(rgb, pattern, window=7):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ok, corners = cv2.findChessboardCornersSB(grey, pattern, cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)  ### newer finder, sub-pixel already
    if ok:
        return corners.reshape(-1, 2)
    ok, corners = cv2.findChessboardCorners(grey, pattern)  ### week 3 finder as the fallback
    if not ok:
        return None
    corners = cv2.cornerSubPix(grey, corners, (window, window), (-1, -1), CRITERIA)
    return corners.reshape(-1, 2)


### the board looks the same turned 180 deg, keep the corner order the same as the reference image
def same_way_round(corners, reference, pattern):
    row = corners[pattern[0] - 1] - corners[0]
    reference_row = reference[pattern[0] - 1] - reference[0]
    return corners if row @ reference_row > 0 else corners[::-1].copy()  ### reversed list = grid turned 180 deg


### zhang's method: K, distortion, the board pose in each view and the rms error
def calibrate(object_points, image_points, size, flags=cv2.CALIB_FIX_K3):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [object_points] * len(image_points), [p.astype(np.float32) for p in image_points], size, None, None,
        flags=flags)  ### k3 fixed at 0, standard unless the lens is wide angle
    return rms, K, dist.ravel(), rvecs, tvecs


### per view corner errors (px), week 3 part c
def reprojection_errors(object_points, image_points, K, dist, rvecs, tvecs):
    errors = []
    for points, rvec, tvec in zip(image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
        errors.append(np.linalg.norm(points - projected.reshape(-1, 2), axis=1))
    return errors


### camera pose in the tool frame (AX = XB), from the arm's fk and the board pose in each view
def hand_eye(tool_poses, rvecs, tvecs):
    rot_tool = [T.R for T in tool_poses]
    pos_tool = [T.t for T in tool_poses]
    rot_board = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]  ### rotation vectors -> matrices
    pos_board = [tvec.ravel() for tvec in tvecs]
    rot, pos = cv2.calibrateHandEye(rot_tool, pos_tool, rot_board, pos_board)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos.ravel()
    return T


### numbers and arrays to json
def save(path, **values):
    with open(path, "w") as file:
        json.dump({key: np.asarray(value).tolist() if isinstance(value, (np.ndarray, list, tuple)) else value
                   for key, value in values.items()}, file, indent=2)


### json back to numbers and arrays
def load(path):
    with open(path) as file:
        values = json.load(file)
    return {key: np.asarray(value) if isinstance(value, list) else value for key, value in values.items()}