import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # figures go to file, the mujoco viewer keeps the screen
import matplotlib.pyplot as plt

from math import degrees

from assets.objects import Basket, Table, Target
from assets.ur3 import UR3
from assets.camera import SURVEY_ANGLES, look_down_pose, rgbd, tool_from_camera
from classes.movement import kinematics, solve_ikine_with_limits
from src.perception.calibration import load
from src.perception.localise import jar_locator


ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

table = Table()
target = Target(table, seed=3)
basket = Basket(table)
ur3 = UR3(base=table.top_centre(), scene=[table, target, basket])
ur3.add_to_env()
camera = rgbd(ur3.model, ur3.data, seed=0)
kin = kinematics(ur3)

mount = table.top_centre().t
roi = (mount + [0.0, -0.50, -0.03], mount + [0.55, 0.50, 0.25])  # the workspace box on the table, as in week 4 part 4
locator = jar_locator(load(os.path.join(ROOT, "assets", "calibration.json")), roi)  # calibrated K, dist and hand-eye only

# the three search configurations, solved once so the arm can jump between them
survey_q, q = [], np.asarray(UR3.home_q, dtype=float)
for angle in SURVEY_ANGLES:
    q = solve_ikine_with_limits.solve(kin, tool_from_camera(look_down_pose(mount, angle)), q)
    survey_q.append(q.copy())


def true_lid():  # ground truth, scoring only
    return target.pos + np.array([0.0, 0.0, target.height])


def look(q):  # put the arm in a search configuration and run perception
    ur3.q = list(q)
    ur3.update()
    rgb, depth = camera.read()
    return rgb, depth, locator.locate(depth, kin.fkine(ur3.q))


# ---- part A: one jar, the three views, week 4 style figures ----
print("jar (true) lid at", np.round(true_lid(), 4))
for index, (angle, q) in enumerate(zip(SURVEY_ANGLES, survey_q)):
    rgb, depth, found = look(q)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].imshow(rgb)
    ax[0].set_title(f"colour image, view {degrees(angle):+.0f} deg")
    im = ax[1].imshow(np.where(depth > 0, depth, np.nan), cmap="viridis")
    ax[1].set_title("noisy depth [m], white = no return")
    plt.colorbar(im, ax=ax[1], shrink=0.8)
    if found is None:
        print(f"view {degrees(angle):+.0f} deg: no jar")
        ax[2].set_title("no jar found")
    else:
        error = (found["lid"] - true_lid()) * 1000
        sigma = np.sqrt(np.diag(found["R"])) * 1000
        print(f"view {degrees(angle):+.0f} deg: lid {np.round(found['lid'], 4)}, error {np.round(error, 1)} mm, "
              f"sigma {np.round(sigma, 1)} mm, {found['count']} points, width {found['width'] * 1000:.0f} mm, "
              f"height {found['height'] * 1000:.0f} mm")
        cluster = found["cluster"]
        ax[2].scatter(cluster[:, 0], cluster[:, 1], c=cluster[:, 2], s=4, cmap="viridis")
        ax[2].plot(*found["lid"][:2], "r+", ms=16, mew=2, label="estimate")
        ax[2].plot(*true_lid()[:2], "kx", ms=12, mew=2, label="truth")
        ax[2].set_aspect("equal")
        ax[2].legend()
        ax[2].set_title("jar cluster, top view (world x, y)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, f"perception_view_{index}.png"), dpi=110)
    plt.close(fig)

# ---- part B: many random jar positions, how good is a single measurement ----
trials = 50
errors, normalised, missed = [], [], 0
for trial in range(trials):
    target.set_pose(ur3.model, ur3.data, target.sample())  # new random position, moved directly in the sim
    best = None
    for q in survey_q:
        _, _, found = look(q)
        if found is not None and (best is None or found["count"] > best["count"]):
            best = found
    if best is None:
        missed += 1
        continue
    error = best["lid"] - true_lid()
    errors.append(error)
    normalised.append(error / np.sqrt(np.diag(best["R"])))  # error in units of its own sigma

errors, normalised = np.array(errors) * 1000, np.array(normalised)
print(f"\n{trials - missed} of {trials} random jars found")
print(f"mean error (mm)  x {errors[:, 0].mean():+.1f}  y {errors[:, 1].mean():+.1f}  z {errors[:, 2].mean():+.1f}")
print(f"rms error (mm)   x {np.sqrt((errors[:, 0] ** 2).mean()):.1f}  y {np.sqrt((errors[:, 1] ** 2).mean()):.1f}  "
      f"z {np.sqrt((errors[:, 2] ** 2).mean()):.1f}")
print(f"worst xy error {np.max(np.hypot(errors[:, 0], errors[:, 1])):.1f} mm")
print(f"errors within 2 sigma: x {np.mean(np.abs(normalised[:, 0]) < 2) * 100:.0f} %  "
      f"y {np.mean(np.abs(normalised[:, 1]) < 2) * 100:.0f} %  z {np.mean(np.abs(normalised[:, 2]) < 2) * 100:.0f} %  "
      f"(about 95 % if R is honest)")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for a, axis in zip(ax, "xyz"):
    column = "xyz".index(axis)
    a.hist(errors[:, column], bins=15)
    a.axvline(0, color="k", lw=1)
    a.set_xlabel(f"{axis} error [mm]")
    a.set_title(f"lid {axis}: mean {errors[:, column].mean():+.1f}, rms {np.sqrt((errors[:, column] ** 2).mean()):.1f} mm")
fig.suptitle(f"single-frame jar localisation error over {len(errors)} random placements")
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "perception_errors.png"), dpi=110)
plt.close(fig)
print("figures saved to", FIGURE_DIR)

camera.close()
ur3.hold()  # keep the viewer open, close the window to finish
ur3.close()