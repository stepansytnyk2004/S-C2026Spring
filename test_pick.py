import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from assets.objects import Basket, Table, Target
from assets.ur3 import UR3
from assets.camera import rgbd
from assets.joints import CONTROL_DT, joint_plant
from src.control.lqr import joint_controller
from src.control.lqg import arm_lqg
from src.control.pick import pick_task
from src.estimation.observer import joint_observer
from src.estimation.target_filter import target_filter
from src.perception.calibration import load
from src.perception.localise import jar_locator


ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

trial = sys.argv[1] if len(sys.argv) > 1 else "nominal"  
seed = int(sys.argv[2]) if len(sys.argv) > 2 else None    

table = Table()
target = Target(table, seed=seed)
basket = Basket(table)
ur3 = UR3(base=table.top_centre(), scene=[table, target, basket])
ur3.add_to_env()
camera = rgbd(ur3.model, ur3.data, seed=seed)
if trial == "noise":
    camera.noise_scale = 3.0  

plant = joint_plant(UR3.home_q, seed=seed)
observer = joint_observer(UR3.home_q, CONTROL_DT)
controller = joint_controller(CONTROL_DT)
arm = arm_lqg(ur3, plant, observer, controller)

mount = table.top_centre().t
roi = (mount + [0.0, -0.50, -0.03], mount + [0.55, 0.50, 0.25])
locator = jar_locator(load(os.path.join(ROOT, "assets", "calibration.json")), roi)

truth = []  
nudged = {}


def on_frame(time):  
    if trial == "nudge" and "at" not in nudged and task.events and task.events[-1][1] == "pbvs approach" \
            and time > task.events[-1][0] + 1.5:
        target.set_pose(ur3.model, ur3.data, target.pos + np.array([0.03, -0.03, 0.0])) 
        nudged["at"] = time
        print(f"   >>> target nudged 4 cm at {time:.2f} s")
    truth.append(target.pos + np.array([0.0, 0.0, target.height]))


task = pick_task(arm, camera, locator, target_filter(), mount, basket, UR3.home_q, on_frame=on_frame)
reach = {}


def grip():  
    pos, _ = ur3.holder.tool_pose()
    reach["error"] = (pos - (target.pos + np.array([0.0, 0.0, target.height]))) * 1000
    return ur3.holder.grip(target)


result = task.run(grip=grip, release=lambda floor: ur3.holder.release(floor))

# ---- evaluation ----
in_basket = bool(np.all(np.abs(target.pos[:2] - basket.pos[:2]) < basket.half))
letters = {"start": "s", "accepted": "a", "rejected": "x", "restart": "R", "none": "-"}
print(f"\n{trial} trial: found {result['found']}, pbvs converged {result['converged']}, picked {result['picked']}, "
      f"in the basket {in_basket}")
print("frames:", "".join(letters[f["status"]] for f in task.frames), "(s start, a accepted, x rejected, R restart, - none)")
if "error" in reach:
    print(f"tool to true lid at the grip (mm): {np.round(reach['error'], 1)}")

frames = [f for f in task.frames if f["estimate"] is not None]
if not frames:  # nothing was detected so nothing to plot
    print("no target estimate to plot (was the viewer window closed early?)")
    camera.close()
    ur3.close()
    sys.exit()
t = np.array([f["t"] for f in frames])
estimate = np.array([f["estimate"] for f in frames]) * 1000
sigma = np.array([f["sigma"] for f in frames]) * 1000
true = np.array([truth[i] for i, f in enumerate(task.frames) if f["estimate"] is not None]) * 1000
servo_start = next(time for time, what in task.events if what == "pbvs approach") if result["found"] else None

fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
for axis, a in enumerate(ax):
    a.fill_between(t, estimate[:, axis] - 2 * sigma[:, axis], estimate[:, axis] + 2 * sigma[:, axis],
                   step="post", color="C0", alpha=0.25, label="estimate $\\pm 2\\sigma$")
    a.step(t, true[:, axis], "k", where="post", lw=1.5, label="true")
    a.step(t, estimate[:, axis], "C0", where="post", lw=1.2, label="estimate")
    for status, colour in (("rejected", "C3"), ("restart", "C1")):
        picked = [i for i, f in enumerate(frames) if f["status"] == status]
        a.plot(t[picked], [frames[i]["z"][axis] * 1000 for i in picked], "o", color=colour, ms=5,
               label=f"detection {status}")
    if servo_start is not None:
        a.axvline(servo_start, color="0.6", ls=":", label="pbvs starts")
    if "at" in nudged:
        a.axvline(nudged["at"], color="C3", ls="--", label="target nudged")
    a.set_ylabel(f"lid {'xyz'[axis]} [mm]")
ax[0].legend(fontsize=7, ncol=4, loc="upper left")
ax[0].set_title(f"target estimate against truth during the pick ({trial} trial)")
ax[2].set_xlabel("time [s]")
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, f"pick_estimate_{trial}.png"), dpi=110)
plt.close(fig)

servo = [f for f in task.frames if f["error"] is not None]
if servo:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot([f["t"] for f in servo], [f["error"] * 1000 for f in servo], "o-", ms=3)
    ax[0].set_ylabel("position error |t - t*| [mm]")
    ax[1].plot([f["t"] for f in servo], [np.degrees(f["angle"]) for f in servo], "o-", ms=3, color="C1")
    ax[1].set_ylabel("rotation error [deg]")
    for a in ax:
        a.set_yscale("log")
        a.set_xlabel("time [s]")
        a.grid(alpha=0.3)
        if "at" in nudged:
            a.axvline(nudged["at"], color="C3", ls="--")
    fig.suptitle(f"pbvs convergence ({trial} trial)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, f"pick_pbvs_{trial}.png"), dpi=110)
    plt.close(fig)
print("figures saved to", FIGURE_DIR)

camera.close()
ur3.hold()  
ur3.close()