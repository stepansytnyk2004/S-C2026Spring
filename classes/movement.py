import time
import mujoco
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3, SO3
from math import pi


TOOL_BODY = "tool0"     # body whose frame counts as the end effector
CHECK_LIMITS = True     # reject a trajectory that leaves the joint ranges
REALTIME = True         # hold the simulation back to wall-clock speed, set False to run flat out


class EStop(Exception):
    pass


class kinematics():
    def __init__(self, robot, tool=TOOL_BODY):
        self.robot = robot
        self.model = robot.model
        self.data = mujoco.MjData(robot.model)  # scratch copy for planning
        self.name = getattr(robot, "name", "robot")

        self.tool = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, tool)
        if self.tool < 0:
            raise ValueError(f"{self.name} has no body named {tool}")

        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in robot.joint_names]
        self.qadr = [self.model.jnt_qposadr[j] for j in joints]  # where each angle lives in qpos
        self.vadr = [self.model.jnt_dofadr[j] for j in joints]   # and its column in the Jacobian
        self.n = len(joints)

        lower, upper = [], []
        for j in joints:  # an unlimited hinge reads as +/- infinity
            limited = bool(self.model.jnt_limited[j])
            lower.append(self.model.jnt_range[j][0] if limited else -np.inf)
            upper.append(self.model.jnt_range[j][1] if limited else np.inf)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)

    def forward(self, q):  # push q through the scratch model
        self.data.qpos[self.qadr] = np.asarray(q, dtype=float).reshape(-1)
        mujoco.mj_forward(self.model, self.data)

    def fkine(self, q):  # tool pose in the world frame
        self.forward(q)
        return SE3.Rt(self.data.xmat[self.tool].reshape(3, 3), self.data.xpos[self.tool])

    def jacob0(self, q):  # 6 x n world-frame Jacobian at the tool origin
        self.forward(q)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.tool)
        return np.vstack([jacp[:, self.vadr], jacr[:, self.vadr]])

    def in_limits(self, q, tol=1e-10):
        q = np.asarray(q, dtype=float)
        return bool(np.all(np.isfinite(q)) and np.all(q >= self.lower - tol) and np.all(q <= self.upper + tol))

    def pose_error(self, q, pose, weights=None):
        current = self.fkine(q)
        position_error = pose.t - current.t
        rotation_error = SO3(pose.R @ current.R.T, check=False).log(twist=True)
        error = np.r_[position_error, rotation_error]
        return error if weights is None else error * weights


class solve_ikine_with_limits():
    def __init__(self, kin, pose, q_start, mask=None, tries=20):
        self.kin = kin
        self.pose = pose
        self.q_start = q_start
        self.mask = mask
        self.tries = tries

    def solve(kin, pose, q_start, mask=None, tries=20, iterations=300, damping=0.05):
        q_start = np.asarray(q_start, dtype=float)
        lower, upper = kin.lower, kin.upper
        weights = np.ones(6) if mask is None else np.asarray(mask, dtype=float)
        rng = np.random.default_rng(0)

        seeds = [q_start.copy()]
        midpoint = q_start.copy()
        bounded_joints = np.isfinite(lower) & np.isfinite(upper)
        midpoint[bounded_joints] = (lower[bounded_joints] + upper[bounded_joints]) / 2
        seeds.append(midpoint)

        # generate seeds
        while len(seeds) < tries:
            seed = np.empty(kin.n, dtype=float)
            for joint in range(kin.n):
                if np.isfinite(lower[joint]) and np.isfinite(upper[joint]):
                    seed[joint] = rng.uniform(lower[joint], upper[joint])
                else:
                    seed[joint] = q_start[joint] + rng.uniform(-2 * pi, 2 * pi)
            seeds.append(seed)

        for seed in seeds:
            q = np.clip(np.asarray(seed, dtype=float).copy(), lower, upper)

            for _ in range(iterations):
                error = kin.pose_error(q, pose, weights)
                if np.linalg.norm(error[:3]) <= 1e-5 and np.linalg.norm(error[3:]) <= 1e-4:
                    break
                J = weights[:, None] * kin.jacob0(q)
                qdot = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), error)
                q = np.clip(q + qdot, lower, upper)  # clamped IK, a step never leaves the ranges

            error = kin.pose_error(q, pose, weights)
            if np.linalg.norm(error[:3]) > 1e-4 or np.linalg.norm(error[3:]) > 1e-3:
                continue  # this seed did not converge
            if not kin.in_limits(q):
                continue

            # prefer the branch the wrist can reach without unwinding, where that is in range
            for joint in range(kin.n):
                turns = np.rint((q_start[joint] - q[joint]) / (2 * pi))
                min_turns = np.ceil((lower[joint] - q[joint] - 1e-10) / (2 * pi))
                max_turns = np.floor((upper[joint] - q[joint] + 1e-10) / (2 * pi))
                if min_turns <= max_turns:
                    q[joint] += 2 * pi * np.clip(turns, min_turns, max_turns)
            if not kin.in_limits(q):
                continue

            return q

        return None


class move_robot():
    def __init__(self, robot, q_target=None, duration=5.0, step=0.02, ikine=None,
                 mask=None, mode="joint"):
        self.robot = robot
        self.q_target = q_target
        self.duration = duration
        self.step = step
        self.ikine = ikine
        self.mask = mask
        self.mode = mode

    def solve(robot, q_target=None, duration=5.0, step=0.02, ikine=None, mask=None,
              mode="joint"):  # general joint translate function for the arm
        for _ in move_robot.steps(
            robot, q_target=q_target, duration=duration, step=step,
            ikine=ikine, mask=mask, mode=mode,
        ):
            pass

    def steps(robot, q_target=None, duration=5.0, step=0.02, ikine=None, mask=None,
              mode="joint"):
        if duration <= 0 or step <= 0:
            raise ValueError("duration and step must be positive")
        if mode not in ("joint", "cartesian"):
            raise ValueError("mode must be 'joint' or 'cartesian'")
        if mode == "cartesian" and ikine is None:
            raise ValueError("mode 'cartesian' requires an ikine target pose")
        if ikine is None and q_target is None:
            raise ValueError("give either q_target or ikine")

        kin = kinematics(robot)
        q_start = np.asarray(robot.q, dtype=float)  # init joint pos
        samples = round(duration / step) + 1

        if mode == "joint":
            if ikine is None:
                q_target = np.asarray(q_target, dtype=float)  # target joint pos
            else:
                q_target = solve_ikine_with_limits.solve(kin, ikine, q_start, mask=mask)
                if q_target is None:
                    raise RuntimeError(
                        f"Could not find a joint-limit-valid {kin.name} configuration "
                        f"for target end-effector pose {ikine} after trying alternate IK seeds"
                    )
            trajectory = rtb.jtraj(q_start, q_target, samples).q  # gen joint space quintic traj

        else:  # cartesian
            trajectory = []
            q = q_start.copy()
            traj_poses = rtb.ctraj(kin.fkine(q_start), ikine, samples)  # gen cartesian space traj
            for pose_index, pose in enumerate(traj_poses):
                q = solve_ikine_with_limits.solve(kin, pose, q, mask=mask, tries=4)
                if q is None:
                    raise RuntimeError(
                        f"Could not find a joint-limit-valid {kin.name} configuration "
                        f"at Cartesian path sample {pose_index + 1}/{len(traj_poses)} "
                        "after trying alternate IK seeds"
                    )
                trajectory.append(q)
            q_target = trajectory[-1].copy()

        # Reject the whole trajectory before moving if any sampled joint angle exceeds its limits.
        if CHECK_LIMITS:
            for step_index, q in enumerate(trajectory):
                below = np.isfinite(kin.lower) & (q < kin.lower - 1e-10)
                above = np.isfinite(kin.upper) & (q > kin.upper + 1e-10)
                invalid_joints = np.flatnonzero(below | above)
                if invalid_joints.size:
                    joint_index = int(invalid_joints[0])
                    raise RuntimeError(
                        f"{kin.name} joint {joint_index + 1} reaches "
                        f"{np.rad2deg(q[joint_index]):.2f} degrees at trajectory step "
                        f"{step_index}; limits are "
                        f"[{np.rad2deg(kin.lower[joint_index]):.2f}, "
                        f"{np.rad2deg(kin.upper[joint_index]):.2f}] degrees"
                    )

        print(f"[{kin.name}] Starting joint angles:", np.round(q_start, 3))
        print(f"[{kin.name}] Target joint angles:  ", np.round(q_target, 3))
        print(f"[{kin.name}] Tip position at start (m):", np.round(kin.fkine(q_start).t, 3))    # FK for the initial tip pos
        print(f"[{kin.name}] Tip position at target (m):", np.round(kin.fkine(q_target).t, 3))  # FK for the target tip pos

        robot.model.opt.timestep = step
        sim_start = float(robot.data.time)
        wall_start = time.perf_counter()    # time

        yield 0.0
        for step_index, q in enumerate(trajectory):  # walk each joint step in the trajectory
            if robot.viewer is not None and not robot.viewer.is_running():
                raise EStop(f"[{kin.name}] viewer closed mid-move")
            robot.q = np.asarray(q, dtype=float).copy()
            robot.update()  # writes qpos, runs FK, redraws
            robot.data.qvel[:] = 0.0  # the pose is commanded, not integrated, so nothing carries over
            dt = step if step_index < len(trajectory) - 1 else 0.0  # samples span one interval fewer
            if dt:
                mujoco.mj_step(robot.model, robot.data)  # advances data.time by one timestep
            if REALTIME:
                lag = (float(robot.data.time) - sim_start) - (time.perf_counter() - wall_start)
                if lag > 0:
                    time.sleep(lag)
            yield dt

        true_ee_pos = kin.fkine(robot.q)
        target_ee_pos = ikine if ikine is not None else kin.fkine(q_target)
        loss_pos = target_ee_pos.t - true_ee_pos.t
        loss_rot = ((target_ee_pos.rpy() - true_ee_pos.rpy()) + np.pi) % (2 * np.pi) - np.pi
        if mask is not None:
            loss_pos = loss_pos * np.asarray(mask, dtype=float)[:3]
            loss_rot = loss_rot * np.asarray(mask, dtype=float)[3:]
        print(f"""[{kin.name}] Move completed in {float(robot.data.time) - sim_start:.2f} s of simulated time.
        Loss:
            Position (m): Mag: {np.linalg.norm(loss_pos):.6f}, X, Y, Z: {np.round(loss_pos, 6)}
            Rotation (rad): Mag: {np.linalg.norm(loss_rot):.6f}, Roll, Pitch, Yaw: {np.round(loss_rot, 6)}"""
        )


class fine_interpolation():
    def solve(q1, q2, max_step_radians=np.deg2rad(3)):
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        steps = int(np.ceil(1.9 * np.max(np.abs(q2 - q1)) / max_step_radians)) + 1
        return rtb.jtraj(q1, q2, max(steps, 2)).q


class interpolate_waypoint():
    def solve(waypoints, max_step_radians=np.deg2rad(3)):
        q_matrix = []
        for i in range(len(waypoints) - 1):
            for q in fine_interpolation.solve(waypoints[i], waypoints[i + 1], max_step_radians):
                q_matrix.append(q)
        return q_matrix
