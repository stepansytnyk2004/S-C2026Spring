import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2


### what is saved for every trial
COLUMNS = ("trial", "disturbance", "start_x", "start_y", "found", "converged", "picked", "in_basket",
           "grip_x_mm", "grip_y_mm", "grip_z_mm", "estimate_error_mm", "target_nees", "rejected", "restarts", "attempts",
           "search_s", "approach_s", "pick_place_s", "total_s", "velocity_rms", "joint_nees")


### collects the trials, prints them and writes the summary, csv and figures
class results:
    def __init__(self, table_dir, figure_dir):
        self.table_dir = table_dir
        self.figure_dir = figure_dir
        self.rows = []
        os.makedirs(table_dir, exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)

    ### one finished trial
    def add(self, **row):
        self.rows.append(row)
        return row

    ### one line for the pause between trials
    def line(self, row):
        outcome = "PICKED and placed" if row["in_basket"] else ("picked, not in basket" if row["picked"] else "FAILED")
        grip = np.hypot(row["grip_x_mm"], row["grip_y_mm"])
        return (f"trial {row['trial']:3d} [{row['disturbance']:5s}] {outcome}: grip error xy {grip:.1f} mm, "
                f"z {row['grip_z_mm']:+.1f} mm, target nees {row['target_nees']:.2f}, "
                f"rejected {row['rejected']}, searches {row['attempts']}, time {row['total_s']:.1f} s")

    ### summary table over all trials, and per disturbance
    def summary(self):
        lines = [f"{'disturbance':12s} {'trials':>6s} {'success':>8s} {'grip xy rms':>12s} {'grip z rms':>11s} "
                 f"{'mean nees':>10s} {'mean time':>10s}"]
        groups = [("all", self.rows)] + [(name, [r for r in self.rows if r["disturbance"] == name])
                                         for name in sorted({r["disturbance"] for r in self.rows})]
        for name, rows in groups:
            if not rows:
                continue
            done = [r for r in rows if r["picked"]]
            success = np.mean([r["in_basket"] for r in rows]) * 100
            xy = np.sqrt(np.mean([r["grip_x_mm"] ** 2 + r["grip_y_mm"] ** 2 for r in done])) if done else np.nan
            z = np.sqrt(np.mean([r["grip_z_mm"] ** 2 for r in done])) if done else np.nan
            nees = np.nanmean([r["target_nees"] for r in rows])
            time = np.mean([r["total_s"] for r in rows])
            lines.append(f"{name:12s} {len(rows):6d} {success:7.0f}% {xy:9.1f} mm {z:8.1f} mm {nees:10.2f} {time:8.1f} s")
        valid = [r["target_nees"] for r in self.rows if np.isfinite(r["target_nees"])]
        if valid:
            lower, upper = chi2.ppf([0.025, 0.975], 3 * len(valid)) / len(valid)
            lines.append(f"target filter nees: mean {np.mean(valid):.2f}, expect 3, 95 % bounds {lower:.2f} to {upper:.2f}")
        joint = [r["joint_nees"] for r in self.rows if np.isfinite(r["joint_nees"])]
        if joint:
            lines.append(f"joint observer: mean nees {np.mean(joint):.2f} (expect 2), "
                         f"velocity rms error {np.mean([r['velocity_rms'] for r in self.rows]):.3f} rad/s")
        return "\n".join(lines)

    ### csv, summary text and figures
    def save(self):
        if not self.rows:
            return
        with open(os.path.join(self.table_dir, "trials.csv"), "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: row[key] for key in COLUMNS})
        with open(os.path.join(self.table_dir, "summary.txt"), "w") as file:
            file.write(self.summary() + "\n")
        self.figures()

    def figures(self):
        rows = self.rows
        colours = {"none": "C0", "nudge": "C3", "noise": "C1"}
        number = np.array([r["trial"] for r in rows])

        ### grip error per trial, xy and z
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for name, colour in colours.items():
            picked = [r for r in rows if r["disturbance"] == name and r["picked"]]
            if picked:
                ax[0].plot([r["grip_x_mm"] for r in picked], [r["grip_y_mm"] for r in picked], "o", color=colour, label=name)
                ax[1].plot([r["trial"] for r in picked], [r["grip_z_mm"] for r in picked], "o", color=colour, label=name)
        ax[0].add_patch(plt.Circle((0, 0), 12, fill=False, ls="--", color="k", label="grip tolerance"))
        ax[0].set_aspect("equal")
        ax[0].set_xlabel("x error [mm]")
        ax[0].set_ylabel("y error [mm]")
        ax[0].set_title("tool to true lid at the grip, top view")
        ax[0].legend(fontsize=8)
        ax[1].axhspan(-10, 10, color="0.9", label="grip tolerance")
        ax[1].set_xlabel("trial")
        ax[1].set_ylabel("z error [mm]")
        ax[1].set_title("tool to true lid at the grip, height")
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_dir, "results_grip_error.png"), dpi=110)
        plt.close(fig)

        ### target filter consistency over the trials
        nees = np.array([r["target_nees"] for r in rows])
        valid = np.isfinite(nees)
        if valid.any():
            lower, upper = chi2.ppf([0.025, 0.975], 3)
            fig, ax = plt.subplots(figsize=(10, 3.5))
            for name, colour in colours.items():
                pick = np.array([r["disturbance"] == name for r in rows]) & valid
                ax.plot(number[pick], nees[pick], "o", color=colour, label=name)
            ax.axhline(3, color="k", lw=1, label="expected (3)")
            ax.axhspan(lower, upper, color="C2", alpha=0.15, label="95 % for one trial")
            ax.set_yscale("log")
            ax.set_xlabel("trial")
            ax.set_ylabel("nees")
            ax.set_title(f"target filter at the grip: mean nees {nees[valid].mean():.2f}")
            ax.legend(fontsize=8)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_dir, "results_nees.png"), dpi=110)
            plt.close(fig)

        ### where the time goes in each trial
        fig, ax = plt.subplots(figsize=(10, 3.5))
        bottom = np.zeros(len(rows))
        for key, label in (("search_s", "search"), ("approach_s", "pbvs approach"), ("pick_place_s", "pick, place, home")):
            values = np.array([r[key] for r in rows])
            ax.bar(number, values, bottom=bottom, label=label)
            bottom += values
        ax.set_xlabel("trial")
        ax.set_ylabel("time [s]")
        ax.set_title("time per trial")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_dir, "results_time.png"), dpi=110)
        plt.close(fig)