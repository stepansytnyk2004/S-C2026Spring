import argparse
import os
import time
import numpy as np

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
from src.perception.camera_view import camera_view

from src.evaluation.disturbance import DISTURBANCES, NOISE_SCALE, nudger
from src.evaluation.groundtruth import grip_error, in_basket, joint_observer_stats, target_nees, true_lid
from src.evaluation.results import results


### where things are read from and written to
ROOT = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(ROOT, "assets", "calibration.json")
FIGURE_DIR = os.path.join(ROOT, "docs", "figures")
RESULTS_DIR = os.path.join(ROOT, "results")


### command line options
def options():
    parser = argparse.ArgumentParser(description="UR3 vision-guided pick and place, trial after trial")
    parser.add_argument("--trials", type=int, default=0, help="number of trials, 0 keeps going until the viewer is closed")
    parser.add_argument("--seed", type=int, default=None, help="seed for repeatable target positions and noise")
    parser.add_argument("--disturb", choices=DISTURBANCES + ("mix",), default="none",
                        help="disturbance in every trial, mix cycles none, nudge, noise")
    parser.add_argument("--pause", type=float, default=2.0, help="pause between trials to read the results (s)")
    parser.add_argument("--headless", action="store_true", help="no viewer, runs as fast as it can")
    parser.add_argument("--camera", action="store_true", help="window with the wrist camera, detections, filter estimate and pbvs errors")
    return parser.parse_args()


### true while the viewer is open, always true without one
def running(ur3, headless):
    return headless or (ur3.viewer is not None and ur3.viewer.is_running())


### hold still for a while with the viewer responsive
def pause(ur3, seconds, headless):
    if headless:
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end and running(ur3, headless):
        ur3.sync()
        time.sleep(0.02)


### time spent searching, approaching, and picking, placing and going home
def phase_times(events, began, ended):
    found = next((t for t, what in events if what.startswith("target found") or what == "target not found"), ended)
    servo = next((t for t, what in reversed(events) if what.startswith("pbvs converged") or what == "pbvs timed out"), found)
    return found - began, servo - found, ended - servo


def main():
    args = options()
    if args.headless and args.trials == 0:
        args.trials = 20
    rng = np.random.default_rng(args.seed)

    ### the simulator: table, target, basket, the arm with its wrist camera, the true joint drives
    table = Table()
    target = Target(table, seed=args.seed)
    basket = Basket(table)
    ur3 = UR3(base=table.top_centre(), scene=[table, target, basket])
    if not args.headless:
        ur3.add_to_env()
    camera = rgbd(ur3.model, ur3.data, seed=args.seed)
    plant = joint_plant(UR3.home_q, seed=args.seed)

    ### the robot's side: joint observer and lqr, perception from the calibration file only
    observer = joint_observer(UR3.home_q, CONTROL_DT)
    controller = joint_controller(CONTROL_DT)
    arm = arm_lqg(ur3, plant, observer, controller, realtime=not args.headless)
    mount = table.top_centre().t
    roi = (mount + [0.0, -0.50, -0.03], mount + [0.55, 0.50, 0.25])  ### the workspace box on the table
    locator = jar_locator(load(CALIBRATION_FILE), roi)
    view = camera_view(arm, camera, locator) if args.camera else None
    if view is not None:
        arm.on_draw = view  ### arm redraws window as it moves

    log = results(RESULTS_DIR, FIGURE_DIR)
    trial = 0
    try:
        while running(ur3, args.headless) and (args.trials == 0 or trial < args.trials):
            trial += 1
            disturbance = DISTURBANCES[(trial - 1) % len(DISTURBANCES)] if args.disturb == "mix" else args.disturb

            ### a new trial: the target appears somewhere new on the table
            start = target.sample()
            target.set_pose(ur3.model, ur3.data, start, [1.0, 0.0, 0.0, 0.0])
            ur3.update()
            camera.noise_scale = NOISE_SCALE if disturbance == "noise" else 1.0
            push = nudger(target, ur3.model, ur3.data, rng) if disturbance == "nudge" else None
            print(f"\n===== trial {trial} [{disturbance}] target at ({start[0]:.3f}, {start[1]:.3f}) =====")

            task = pick_task(arm, camera, locator, target_filter(), mount, basket, UR3.home_q, on_frame=push)
            if push is not None:
                push.task = task
            if view is not None:
                view.task = task
                view.title = f"trial {trial} [{disturbance}]"
            scored = {}

            ### the simulator decides if the tool is on the lid, the evaluation scores how close it was
            def grip():
                scored["error"] = grip_error(ur3.holder, target)
                scored["estimate"] = np.linalg.norm(task.filter.estimate - true_lid(target)) * 1000
                scored["nees"] = target_nees(task.filter.estimate, task.filter.covariance(), target)
                return ur3.holder.grip(target)

            began = arm.time
            result = task.run(grip=grip, release=lambda floor: ur3.holder.release(floor))
            if not running(ur3, args.headless):
                break  ### viewer closed part way, this trial does not count

            search, approach, pick_place = phase_times(task.events, began, arm.time)
            velocity_rms, joint_nees = joint_observer_stats(arm.history())
            arm.reset_log()
            error = scored.get("error", np.full(3, np.nan))
            statuses = [status for _, status, _ in task.filter.log]
            row = log.add(trial=trial, disturbance=disturbance, start_x=start[0], start_y=start[1],
                          found=result["found"], converged=result["converged"], picked=result["picked"],
                          in_basket=result["picked"] and in_basket(target, basket),
                          grip_x_mm=error[0], grip_y_mm=error[1], grip_z_mm=error[2],
                          estimate_error_mm=scored.get("estimate", np.nan), target_nees=scored.get("nees", np.nan),
                          rejected=statuses.count("rejected"), restarts=statuses.count("restart"), attempts=result["attempts"],
                          search_s=search, approach_s=approach, pick_place_s=pick_place, total_s=arm.time - began,
                          velocity_rms=velocity_rms, joint_nees=joint_nees)
            print(log.line(row))
            pause(ur3, args.pause, args.headless)
    except KeyboardInterrupt:
        print("\nstopped")

    ### results over every finished trial
    if log.rows:
        log.save()
        print("\n" + log.summary())
        print(f"\ntrials saved to {os.path.join(RESULTS_DIR, 'trials.csv')}, figures to {FIGURE_DIR}")
    if view is not None:
        view.close()    
    camera.close()
    ur3.close()


if __name__ == "__main__":
    main()