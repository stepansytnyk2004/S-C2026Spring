import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import chi2
from spatialmath import SE3

from assets.objects import Basket, Table, Target
from assets.ur3 import UR3
from assets.camera import SURVEY_ANGLES, look_down_pose, rgbd, tool_from_camera
from classes.movement import kinematics, solve_ikine_with_limits
from src.perception.calibration import load
from src.perception.localise import jar_locator
from src.estimation.target_filter import target_filter


ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

APPROACH_HEIGHTS = (0.32, 0.28, 0.26)  
FRAMES = 2                             

table = Table()
target = Target(table, seed=5)
basket = Basket(table)
ur3 = UR3(base=table.top_centre(), scene=[table, target, basket])
ur3.add_to_env()
camera = rgbd(ur3.model, ur3.data, seed=0)
kin = kinematics(ur3)

mount = table.top_centre().t
roi = (mount + [0.0, -0.50, -0.03], mount + [0.55, 0.50, 0.25])
locator = jar_locator(load(os.path.join(ROOT, "assets", "calibration.json")), roi)  

survey_q, q = [], np.asarray(UR3.home_q, dtype=float)
for angle in SURVEY_ANGLES:
    q = solve_ikine_with_limits.solve(kin, tool_from_camera(look_down_pose(mount, angle)), q)
    survey_q.append(q.copy())


def true_lid(): 
    return target.pos + np.array([0.0, 0.0, target.height])


def look(q):  
    ur3.q = list(q)
    ur3.update()
    _, depth = camera.read()
    return locator.locate(depth, kin.fkine(ur3.q))


def above(estimate, height, q_seed):  
    bearing = np.arctan2(estimate[1] - mount[1], estimate[0] - mount[0])
    pose = SE3(estimate[0], estimate[1], mount[2] + height) * SE3.Rz(bearing + pi / 2) * SE3.Rx(pi)
    return solve_ikine_with_limits.solve(kin, tool_from_camera(pose), q_seed)


def feed(f, found, clock, record): 
    status = f.update(found, clock)
    if record is not None:
        record.append(dict(z=found["lid"].copy(), status=status, estimate=f.estimate,
                           sigma=np.sqrt(np.diag(f.covariance())), truth=true_lid()))


def run_trial(f, clock=0.0, record=None): 
    q_last = survey_q[0]
    for q in survey_q:
        clock += 1.0
        found = look(q)
        if found is not None:
            feed(f, found, clock, record)
            q_last = q
    if not f.ready:
        return clock, q_last
    for height in APPROACH_HEIGHTS:
        q_view = above(f.estimate, height, q_last)
        if q_view is None:
            continue
        q_last = q_view
        for _ in range(FRAMES):
            clock += 0.1
            found = look(q_view)
            if found is not None and np.linalg.norm(found["lid"][:2] - true_lid()[:2]) > 0.01:  # ground truth, debugging only
                rgb, _ = camera.read()
                plt.imsave(os.path.join(FIGURE_DIR, f"approach_off_{np.random.randint(10000)}.png"), rgb)
            if found is not None:
                feed(f, found, clock, record)
    return clock, q_last


f = target_filter()
record = []
clock, q_last = run_trial(f, record=record)
nudge_at = len(record)
target.set_pose(ur3.model, ur3.data, target.pos + np.array([0.03, -0.03, 0.0]))  
for _ in range(6):  
    clock += 0.1
    found = look(q_last)
    if found is not None:
        feed(f, found, clock, record)
for height in APPROACH_HEIGHTS[1:]:  
    q_view = above(f.estimate, height, q_last)
    if q_view is None:
        continue
    q_last = q_view
    for _ in range(FRAMES):
        clock += 0.1
        found = look(q_view)
        if found is not None:
            feed(f, found, clock, record)

print("part a, detection by detection:")
for index, r in enumerate(record):
    error = (r["estimate"] - r["truth"]) * 1000
    print(f"  {index:2d} {r['status']:9s} estimate error {np.round(error, 1)} mm, sigma {np.round(r['sigma'] * 1000, 1)} mm")

steps = np.arange(len(record))
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
colours = {"start": "C0", "accepted": "C2", "rejected": "C3", "restart": "C1"}
for axis, a in enumerate(ax):
    truth = np.array([r["truth"][axis] for r in record]) * 1000
    estimate = np.array([r["estimate"][axis] for r in record]) * 1000
    sigma = np.array([r["sigma"][axis] for r in record]) * 1000
    a.fill_between(steps, estimate - 2 * sigma, estimate + 2 * sigma, step="mid", color="C0", alpha=0.25,
                   label="estimate $\\pm 2\\sigma$")
    a.step(steps, truth, "k", where="mid", lw=1.5, label="true")
    a.step(steps, estimate, "C0", where="mid", lw=1.2, label="estimate")
    for status, colour in colours.items():
        picked = [i for i, r in enumerate(record) if r["status"] == status]
        a.plot(picked, [record[i]["z"][axis] * 1000 for i in picked], "o", color=colour, ms=5, label=f"detection: {status}")
    a.axvline(nudge_at - 0.5, color="0.5", ls="--", label="jar nudged")
    a.set_ylabel(f"lid {'xyz'[axis]} [mm]")
ax[0].legend(fontsize=7, ncol=4, loc="upper left")
ax[0].set_title("jar filter: estimate against truth, with the jar nudged 4 cm")
ax[2].set_xlabel("detection")
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "jar_filter_nudge.png"), dpi=110)
plt.close(fig)


trials = 30
errors, nees, nis, rejected = [], [], [], 0
letters = {"start": "s", "accepted": "a", "rejected": "x", "restart": "R"}
for trial in range(trials):
    target.set_pose(ur3.model, ur3.data, target.sample())
    f = target_filter()
    steps_b = []
    run_trial(f, record=steps_b)
    if not f.ready:
        continue
    error = f.estimate - true_lid()
    first = (steps_b[0]["z"] - steps_b[0]["truth"]) * 1000
    print(f"  {trial:2d} {''.join(letters[r['status']] for r in steps_b):12s} "
          f"first detection error {np.round(first, 1)} mm, final error {np.round(error * 1000, 1)} mm")
    errors.append(error)
    nees.append(float(error @ np.linalg.solve(f.covariance(), error)))
    nis += [n for _, status, n in f.log if status == "accepted"]
    rejected += sum(1 for _, status, _ in f.log if status in ("rejected", "restart"))

errors, nees = np.array(errors) * 1000, np.array(nees)
lower, upper = chi2.ppf([0.025, 0.975], 3 * len(nees)) / len(nees)
print(f"\npart b: {len(nees)} of {trials} jars found")
print(f"final error rms (mm): x {np.sqrt((errors[:, 0] ** 2).mean()):.1f}  y {np.sqrt((errors[:, 1] ** 2).mean()):.1f}  "
      f"z {np.sqrt((errors[:, 2] ** 2).mean()):.1f}, worst xy {np.max(np.hypot(errors[:, 0], errors[:, 1])):.1f}")
print(f"mean nees {nees.mean():.2f} (expect 3, 95 % bounds {lower:.2f} to {upper:.2f})")
print(f"mean nis of accepted detections {np.mean(nis):.2f} (expect 3 if R is honest)")
print(f"detections rejected with the jar standing still: {rejected} (should be 0 or close)")

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(nees, "o", label="final nees per jar")
ax.axhline(3, color="k", lw=1, label="expected (3)")
ax.axhline(nees.mean(), color="C0", ls=":", label=f"mean {nees.mean():.2f}")
ax.axhspan(lower, upper, color="C2", alpha=0.2, label="95 % bounds for the mean")
ax.set_yscale("log")
ax.set_xlabel("random jar")
ax.set_title("is the jar filter consistent?")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "jar_filter_nees.png"), dpi=110)
plt.close(fig)
print("figures saved to", FIGURE_DIR)

camera.close()
ur3.hold()  
ur3.close()