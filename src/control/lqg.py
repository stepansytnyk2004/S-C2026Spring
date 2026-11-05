import time

import numpy as np

from classes.movement import kinematics, solve_ikine_with_limits


### quintic joint trajectory
def quintic(q0, q1, duration, dt):
    t = np.arange(0.0, duration + dt / 2, dt)
    s = t / duration
    step = np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float)
    shape = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    rate = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / duration
    accel = (60 * s - 180 * s ** 2 + 120 * s ** 3) / duration ** 2
    return np.asarray(q0, dtype=float) + np.outer(shape, step), np.outer(rate, step), np.outer(accel, step)


### lqg on the arm
class arm_lqg:
    def __init__(self, robot, plant, observer, controller, display_step=0.02, realtime=True):
        self.robot = robot
        self.plant = plant
        self.observer = observer
        self.controller = controller
        self.dt = plant.dt
        self.display_every = max(1, round(display_step / self.dt))  ### control steps per redraw
        self.realtime = realtime
        self.kin = kinematics(robot)
        self.torque = np.zeros(6)
        self.goal = observer.q
        self.time = 0.0
        self.log = {key: [] for key in ("t", "q_ref", "qd_ref", "q_hat", "qd_hat", "d_hat", "P", "torque",
                                        "q_true", "qd_true", "d_true")}

    ### follow a reference sample by sample
    def track(self, q_ref, qd_ref, qdd_ref):
        start = time.perf_counter()
        for k in range(len(q_ref)):
            if self.robot.viewer is not None and not self.robot.viewer.is_running():
                break
            self.observer.update(self.torque, self.plant.encoders())  ### measure, predict and correct
            self.torque = self.controller.torque(self.observer.q, self.observer.qd, self.observer.d,
                                                 q_ref[k], qd_ref[k], qdd_ref[k])
            self.record(q_ref[k], qd_ref[k])  ### truth and estimate at same instant
            self.plant.step(self.torque)
            self.time += self.dt
            if k % self.display_every == 0:
                self.robot.q = list(self.plant.q)  ### viewer shows where arm is
                self.robot.update()
                if self.realtime:
                    lag = (k + 1) * self.dt - (time.perf_counter() - start)
                    if lag > 0:
                        time.sleep(lag)
        self.robot.q = list(self.plant.q)
        self.robot.update()

    ### joint move from where the observer thinks the arm is
    def move(self, q_target, duration):
        self.goal = np.asarray(q_target, dtype=float)
        self.track(*quintic(self.observer.q, self.goal, duration, self.dt))

    ### stay at the last goal for a while
    def wait(self, duration):
        n = int(round(duration / self.dt))
        self.track(np.tile(self.goal, (n, 1)), np.zeros((n, 6)), np.zeros((n, 6)))

<<<<<<< HEAD
    ### follow joint velocity command 
    def servo(self, qd, duration):
        n = max(1, int(round(duration / self.dt)))
        steps = (np.arange(n) + 1) * self.dt
        q_ref = self.goal + np.outer(steps, qd)  ### reference ramps at qd
        self.track(q_ref, np.tile(qd, (n, 1)), np.zeros((n, 6)))
        self.goal = q_ref[-1].copy()

=======
>>>>>>> 0be77e255e09804efee4c28eb5e8239a2f1a65e5
    ### move the tool to a pose, ik seeded from the estimated joint angles
    def move_to_pose(self, pose, duration, mask=None):
        q_target = solve_ikine_with_limits.solve(self.kin, pose, self.observer.q, mask=mask)
        if q_target is None:
            raise RuntimeError(f"no joint-limit-valid configuration for {pose}")
        self.move(q_target, duration)
        return q_target

    ### the estimates and, beside them, the truth for evaluation only
    def record(self, q_ref, qd_ref):
        self.log["t"].append(self.time)
        self.log["q_ref"].append(np.array(q_ref))
        self.log["qd_ref"].append(np.array(qd_ref))
        self.log["q_hat"].append(self.observer.q)
        self.log["qd_hat"].append(self.observer.qd)
        self.log["d_hat"].append(self.observer.d)
        self.log["P"].append(np.array(self.observer.P))
        self.log["torque"].append(self.torque.copy())
        self.log["q_true"].append(self.plant.q)  ### evaluation only
        self.log["qd_true"].append(self.plant.qd)  ### evaluation only
        self.log["d_true"].append(self.plant.disturbance.copy())  ### evaluation only

    ### the log as arrays
    def history(self):
        return {key: np.array(value) for key, value in self.log.items()}