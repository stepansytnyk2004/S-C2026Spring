import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # figures are saved to file, the mujoco viewer keeps the screen
import matplotlib.pyplot as plt

from math import cos, degrees, pi, sin
from spatialmath import SE3

from assets.objects import Basket, Table, Target
from assets.ur3 import UR3
from assets.camera import CAMERA_POS, CAMERA_SIZE, SURVEY_ANGLES, look_down_pose, project, rgbd, tool_from_camera
from classes.movement import move_robot


FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

table = Table()
target = Target(table)  # random each run
basket = Basket(table)
ur3 = UR3(base=table.top_centre(), scene=[table, target, basket])
ur3.add_to_env()
camera = rgbd(ur3.model, ur3.data)

mount = table.top_centre().t
lid = target.pos + [0, 0, target.height]  # ground truth, only to mark the jar on the images
print("true intrinsics K (evaluation only):\n", np.round(camera.K, 1))
print("jar at", np.round(target.pos, 3))

move_robot.solve(ur3, q_target=UR3.home_q, duration=1.5)  # back to home, wherever it started

for index, angle in enumerate(SURVEY_ANGLES):
    wanted = look_down_pose(mount, angle)
    move_robot.solve(ur3, ikine=tool_from_camera(wanted), duration=2.5)
    rgb, depth = camera.read()

    # checks: the camera ended where it was sent, and the mount maths agrees with mujoco's camera
    pos, rot = ur3.holder.tool_pose()
    from_tool = SE3.Rt(rot, pos, check=False) * SE3(*CAMERA_POS)
    print(f"view {degrees(angle):+.0f} deg: camera vs wanted {np.linalg.norm(camera.pose().t - wanted.t) * 1000:.2f} mm, "
          f"camera vs tool x mount {np.linalg.norm(camera.pose().t - from_tool.t) * 1000:.2f} mm")

    uv, z = camera.project(lid)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].imshow(rgb)
    if z[0] > 0:
        ax[0].plot(uv[0, 0], uv[0, 1], "r+", ms=14, mew=2, label="true lid centre")
        ax[0].legend(loc="upper right")
    ax[0].set_title(f"colour image, view {degrees(angle):+.0f} deg")
    im = ax[1].imshow(np.where(depth > 0, depth, np.nan), cmap="viridis")
    ax[1].set_title("depth image [m], white = no return")
    plt.colorbar(im, ax=ax[1], shrink=0.8)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, f"camera_view_{index}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("   saved", path)

# coverage: over many random jar positions, is the whole jar inside at least one of the three views
width, height = CAMERA_SIZE
views = [look_down_pose(mount, angle) for angle in SURVEY_ANGLES]
rng = np.random.default_rng(1)
seen, trials = 0, 1000
for _ in range(trials):
    radius = rng.uniform(*Target.reach)
    angle = rng.uniform(-Target.half_angle, Target.half_angle)
    base = mount + [radius * cos(angle), radius * sin(angle), 0.0]
    rim = np.array([base + [target.radius * cos(a), target.radius * sin(a), h]
                    for a in np.linspace(0, 2 * pi, 12, endpoint=False) for h in (0.0, target.height)])
    for view in views:
        uv, z = project(camera.K, view, rim)
        if np.all(z > 0) and np.all((uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)):
            seen += 1
            break
print(f"coverage: whole jar inside at least one view for {100 * seen / trials:.1f} % of {trials} random placements")

camera.close()
ur3.hold()  # keep the viewer open, close the window to finish
ur3.close()