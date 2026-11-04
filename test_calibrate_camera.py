import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  ### figures saved to file, the viewer keeps the screen
import matplotlib.pyplot as plt
import cv2
from math import atan, degrees, radians

from assets.objects import Checkerboard, Table
from assets.ur3 import UR3
from assets.camera import CAMERA_POS, CAMERA_SIZE, look_at_from, rgbd, tool_from_camera
from classes.movement import kinematics, move_robot
from src.perception.calibration import calibrate, find_corners, hand_eye, reprojection_errors, same_way_round, save


### output locations
ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(ROOT, "docs", "figures")
CALIBRATION_FILE = os.path.join(ROOT, "assets", "calibration.json")
os.makedirs(FIGURE_DIR, exist_ok=True)

### calibration scene, only the table and the checkerboard
table = Table()
board = Checkerboard(table)
ur3 = UR3(base=table.top_centre(), scene=[table, board])
ur3.add_to_env()
camera = rgbd(ur3.model, ur3.data)
kin = kinematics(ur3)  ### arm's own forward kinematics, for hand-eye

### viewpoints: (distance, tilt, azimuth, roll, x offset, y offset)
views = []
for distance in (0.34, 0.42):
    views += [(distance, 0.0, 0.0, 0.0, 0.0, 0.0), (distance, 0.0, 0.0, radians(15), 0.0, 0.0)]  ### straight down, plain and rolled
    views += [(distance, radians(25), radians(azimuth), radians(10 if azimuth % 180 == 0 else -10), 0.0, 0.0)
              for azimuth in (0, 90, 180, 270)]  ### leaning in from four sides
views += [(0.40, radians(30), radians(azimuth), 0.0, 0.0, 0.0) for azimuth in (45, 135, 225, 315)]  ### leaning in from the diagonals
views += [(0.42, 0.0, 0.0, 0.0, dx, dy) for dx in (-0.07, 0.07) for dy in (-0.05, 0.05)]  ### off centre, board near each image corner

### visit each view, keep the images where the board is found
images, corners, tool_poses = [], [], []
ready = None  ### first good configuration, each view starts from here
move_robot.solve(ur3, q_target=UR3.home_q, duration=1.5)
for number, (distance, tilt, azimuth, roll, dx, dy) in enumerate(views):
    print(f"view {number + 1}/{len(views)}: {distance:.2f} m, tilt {degrees(tilt):.0f} deg, azimuth {degrees(azimuth):.0f} deg")
    if ready is not None:
        move_robot.solve(ur3, q_target=ready, duration=1.0)  ### keeps the ik on the same arm configuration
    try:
        move_robot.solve(ur3, ikine=tool_from_camera(look_at_from(board.centre() + np.array([dx, dy, 0.0]), distance, tilt, azimuth, roll)), duration=2.0)
    except RuntimeError:
        print("   out of reach, skipped")
        continue
    if ready is None:
        ready = list(ur3.q)
    rgb, _ = camera.read()
    found = find_corners(rgb, board.pattern)
    if found is None:
        print("   board not found, skipped")
        plt.imsave(os.path.join(FIGURE_DIR, f"calibration_not_found_{number + 1}.png"), rgb)  ### saved to see what went wrong
        continue
    if corners:
        found = same_way_round(found, corners[0], board.pattern)  ### same corner order as the first image
    images.append(rgb)
    corners.append(found)
    tool_poses.append(kin.fkine(ur3.q))
print(f"\n{len(corners)} of {len(views)} views used")
if len(corners) < 5:
    camera.close()
    ur3.close()
    raise SystemExit("too few views to calibrate, see the calibration_not_found images in docs/figures")

### intrinsic calibration and how good it is
object_points = board.object_points()
rms, K, dist, rvecs, tvecs = calibrate(object_points, corners, CAMERA_SIZE)
errors = reprojection_errors(object_points, corners, K, dist, rvecs, tvecs)
width, height = CAMERA_SIZE
print(f"\nK =\n{np.round(K, 2)}")
print(f"distortion (k1, k2, p1, p2, k3) = {np.round(dist, 4)}")
print(f"horizontal field of view = {degrees(2 * atan(width / (2 * K[0, 0]))):.1f} deg")
print(f"reprojection error: rms {rms:.3f} px, worst view mean {max(e.mean() for e in errors):.3f} px")

### hand-eye, where the camera sits on the tool, then save everything perception needs
T_tool_cam = hand_eye(tool_poses, rvecs, tvecs)
save(CALIBRATION_FILE, K=K, dist=dist, size=CAMERA_SIZE, rms=float(rms), T_tool_cam=T_tool_cam,
     views=len(corners))
print(f"\nhand-eye: camera at {np.round(T_tool_cam[:3, 3] * 1000, 2)} mm in the tool frame")
print("saved", CALIBRATION_FILE)

### comparison with the simulator's true values, evaluation only
K_true = camera.K
print("\n---- against the true values (evaluation only) ----")
print(f"focal length error: fx {100 * (K[0, 0] / K_true[0, 0] - 1):+.3f} %, fy {100 * (K[1, 1] / K_true[1, 1] - 1):+.3f} %")
print(f"principal point error: ({K[0, 2] - K_true[0, 2]:+.2f}, {K[1, 2] - K_true[1, 2]:+.2f}) px")
mount_error = np.linalg.norm(T_tool_cam[:3, 3] - np.asarray(CAMERA_POS)) * 1000
tilt_error = degrees(np.arccos(np.clip((np.trace(T_tool_cam[:3, :3]) - 1) / 2, -1, 1)))  ### rotation angle from identity
print(f"hand-eye error: position {mount_error:.2f} mm, rotation {tilt_error:.3f} deg (true mount {CAMERA_POS}, no rotation)")

### figure: four of the views with their detected corners
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for a, index in zip(ax.ravel(), np.linspace(0, len(images) - 1, 4).astype(int)):
    shown = images[index].copy()
    cv2.drawChessboardCorners(shown, board.pattern, corners[index].reshape(-1, 1, 2).astype(np.float32), True)
    a.imshow(shown)
    a.set_title(f"view {index}, mean error {errors[index].mean():.3f} px")
    a.axis("off")
fig.suptitle("four of the calibration views, detected corners")
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "calibration_views.png"), dpi=110)
plt.close(fig)

### figure: reprojection error per view, week 3 part c
fig = plt.figure(figsize=(8, 4))
plt.bar(range(len(errors)), [e.mean() for e in errors])
plt.axhline(rms, color="r", ls="--", label=f"overall rms {rms:.3f} px")
plt.xlabel("view")
plt.ylabel("mean reprojection error [px]")
plt.title("is the calibration any good? per-view reprojection error")
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "calibration_errors.png"), dpi=110)
plt.close(fig)

### figure: residuals on one image, magnified 50x, week 3 part c
projected, _ = cv2.projectPoints(object_points, rvecs[0], tvecs[0], K, dist)
projected = projected.reshape(-1, 2)
residual = corners[0] - projected
fig = plt.figure(figsize=(8, 6))
plt.imshow(images[0])
plt.plot(corners[0][:, 0], corners[0][:, 1], "g+", ms=8, label="measured")
plt.plot(projected[:, 0], projected[:, 1], "rx", ms=6, label="reprojected")
plt.quiver(corners[0][:, 0], corners[0][:, 1], residual[:, 0], residual[:, 1],
           color="y", angles="xy", scale_units="xy", scale=0.02)
plt.legend()
plt.title("residuals, magnified 50x")
plt.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "calibration_residuals.png"), dpi=110)
plt.close(fig)
print("figures saved to", FIGURE_DIR)

camera.close()
ur3.hold()
ur3.close()