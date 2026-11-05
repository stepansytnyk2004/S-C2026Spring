import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # figures go to file mujoco viewer keeps the screen
import matplotlib.pyplot as plt
from scipy.stats import chi2

from assets.objects import Basket, Table, Target
from assets.ur3 import UR3
from assets.camera import look_down_pose, tool_from_camera
from assets.joints import CONTROL_DT, joint_plant
from src.control.lqr import joint_controller
from src.control.lqg import arm_lqg
from src.estimation.observer import joint_observer


ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

table = Table()
target = Target(table, seed=0)
basket = Basket(table)
ur3 = UR3(base=table.top_centre(), scene=[table, target, basket])
ur3.add_to_env()

plant = joint_plant(UR3.home_q, seed=0)                 # real joints (simulator side)
observer = joint_observer(UR3.home_q, CONTROL_DT)       # kalman observer: encoders -> angle, velocity, load
controller = joint_controller(CONTROL_DT)               # lqr with integral action
arm = arm_lqg(ur3, plant, observer, controller)

for joint, (K, Ki, poles) in enumerate(zip(controller.K, controller.Ki, controller.poles)):
    slowest = float((-CONTROL_DT / np.log(np.abs(poles))).max())
    print(f"joint {joint}: K = {np.round(K, 1)}, Ki = {Ki:.1f}, slowest closed-loop time constant {slowest:.3f} s")


mount = table.top_centre().t
arm.wait(0.5)
arm.move_to_pose(tool_from_camera(look_down_pose(mount, 0.0)), 3.0)
arm.wait(1.0)
load_start = arm.time
plant.disturbance[1:3] = [8.0, 4.0]  
arm.wait(2.0)
load_end = arm.time
plant.disturbance[:] = 0.0
arm.move(UR3.home_q, 3.0)
arm.wait(0.5)


h = arm.history()
t = h["t"]
tracking = (h["q_true"] - h["q_ref"]) * 1000  
velocity_error = h["qd_true"] - h["qd_hat"]
state_error = np.stack([h["q_true"] - h["q_hat"], h["qd_true"] - h["qd_hat"]], axis=2)  
P = h["P"][:, :, :2, :2]
nees = np.einsum("kji,kjil,kjl->kj", state_error, np.linalg.inv(P), state_error)  
nees_mean = nees.mean(axis=1)
lower, upper = chi2.ppf([0.025, 0.975], 2 * 6) / 6  
inside = np.mean((nees_mean > lower) & (nees_mean < upper))
during = (t > load_start + 0.5) & (t < load_end)

print(f"\nmax tracking error (mrad): {np.round(np.abs(tracking).max(axis=0), 2)}")
print(f"velocity estimate rms error (rad/s): {np.round(np.sqrt((velocity_error ** 2).mean(axis=0)), 4)}")
print(f"load estimate during the load (n m): shoulder {h['d_hat'][during, 1].mean():.2f} (true 8.0), "
      f"elbow {h['d_hat'][during, 2].mean():.2f} (true 4.0)")
print(f"peak torque (n m): {np.round(np.abs(h['torque']).max(axis=0), 1)}")
print(f"mean nees {nees.mean():.2f} (expect 2), {inside * 100:.0f} % of steps inside the 95 % bounds")

shown = slice(None, None, 10)  
ts = t[shown]


def shade(ax):
    ax.axvspan(load_start, load_end, color="0.85", label="load on")

fig, ax = plt.subplots(figsize=(10, 4))
shade(ax)
ax.plot(ts, tracking[shown], lw=1)
ax.legend([f"joint {j}" for j in range(6)] + ["load on"], ncol=4, fontsize=8)
ax.set_xlabel("time [s]")
ax.set_ylabel("true angle - reference [mrad]")
ax.set_title("joint tracking error under lqg")
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "lqg_tracking.png"), dpi=110)
plt.close(fig)

joint = 1  
sigma_qd = np.sqrt(h["P"][shown, joint, 1, 1])
sigma_d = np.sqrt(h["P"][shown, joint, 2, 2])
qd_hat, qd_true = h["qd_hat"][shown, joint], h["qd_true"][shown, joint]
d_hat, d_true = h["d_hat"][shown, joint], h["d_true"][shown, joint]
fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
shade(ax[0])
ax[0].fill_between(ts, qd_hat - 2 * sigma_qd, qd_hat + 2 * sigma_qd,
                   color="C0", alpha=0.3, label="estimate $\\pm 2\\sigma$")
ax[0].plot(ts, qd_true, "k", lw=1, label="true")
ax[0].plot(ts, qd_hat, "C0", lw=1, label="estimate")
ax[0].set_ylabel("shoulder velocity [rad/s]")
ax[0].legend(fontsize=8)
ax[0].set_title("unmeasured states: velocity and load torque, estimate against truth")
shade(ax[1])
ax[1].fill_between(ts, d_hat - 2 * sigma_d, d_hat + 2 * sigma_d,
                   color="C1", alpha=0.3, label="estimate $\\pm 2\\sigma$")
ax[1].plot(ts, d_true, "k", lw=1, label="true")
ax[1].plot(ts, d_hat, "C1", lw=1, label="estimate")
ax[1].set_ylabel("shoulder load torque [N m]")
ax[1].set_xlabel("time [s]")
ax[1].legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "lqg_observer.png"), dpi=110)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(ts, nees_mean[shown], lw=0.8, label="nees, mean of 6 joints")
ax.axhline(2, color="k", lw=1, label="expected (2)")
ax.axhline(lower, color="r", ls="--", lw=1, label="95 % bounds")
ax.axhline(upper, color="r", ls="--", lw=1)
ax.set_yscale("log")
ax.set_xlabel("time [s]")
ax.set_title(f"is the joint observer consistent? mean nees {nees.mean():.2f}, {inside * 100:.0f} % inside the bounds")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "lqg_nees.png"), dpi=110)
plt.close(fig)
print("figures saved to", FIGURE_DIR)

ur3.hold()  
ur3.close()