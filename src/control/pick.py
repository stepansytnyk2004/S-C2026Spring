import mujoco
import numpy as np
from math import degrees, pi
from spatialmath import SE3

from assets.camera import SURVEY_ANGLES, look_down_pose, tool_from_camera
from classes.movement import solve_ikine_with_limits


### pbvs
GAIN = 1.0                   ### lambda 
MAX_SPEED = 0.10             ### camera speed limit 
MAX_TURN = 0.5               ### camera turn rate limit 
MAX_JOINT_SPEED = 1.0        ### joint speed limit 
DAMPING = 0.02               ### damped least squares on the arm jacobian
STANDOFF = 0.19              ### camera to lid when the servo ends
FRAME_PERIOD = 0.1          
POSITION_TOLERANCE = 0.003   
ANGLE_TOLERANCE = 0.02       
SERVO_TIMEOUT = 10.0        
LOST_FRAMES = 5              ### frames in a row without a detection before the target counts as lost
RETRIES = 2                  ### fresh searches after the target is lost

### search
SEARCH_POSE = (-0.57, -1.58, 1.27, -1.25, -1.57, 2.57)  
SEARCH_FRAMES = 2           
SETTLE = 0.3                 

### the pick
PICK_SIGMA = 0.006           
PRE_GRASP = 0.04             
LIFT = 0.12                  
DROP_CLEARANCE = 0.03        
SET_DOWN_GAP = 0.002         ### lower the target until it is this close to the basket floor
MOVE_TIME = 2.5             


### heading of search views
def heading(point, mount):
    return np.arctan2(point[1] - mount[1], point[0] - mount[0]) + pi / 2


### rotation matrix -> axis-angle vector, through mujoco quaternions like pose_error in movement.py
def rotation_vector(rot):
    quat, vector = np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot, dtype=float).flatten())
    mujoco.mju_quat2Vel(vector, quat, 1.0)
    return vector


### pose at point
def down(point, yaw):
    return SE3(*point) * SE3.Rz(yaw) * SE3.Rx(pi)


class pick_task:
    def __init__(self, arm, camera, locator, target_filter, mount, basket, home, on_frame=None):
        self.arm = arm                    
        self.camera = camera             
        self.locator = locator              
        self.filter = target_filter         
        self.mount = np.asarray(mount, dtype=float)
        self.basket = basket                
        self.home = np.asarray(home, dtype=float)
        self.on_frame = on_frame         
        self.kin = arm.kin
        self.frames = []                    
        self.events = []       
        self.seen = None            

    def note(self, what):
        self.events.append((self.arm.time, what))
        print(f"[{self.arm.time:6.2f} s] {what}")

    ### camera pose from estimated joint angles/calibrated mount
    def camera_pose(self):
        return self.kin.fkine(self.arm.observer.q) * self.locator.T_tool_cam

    def look(self, phase):
        if self.on_frame is not None:
            self.on_frame(self.arm.time)
        rgb, depth = self.camera.read()
        found = self.locator.locate(depth, self.kin.fkine(self.arm.observer.q))
        self.seen = (rgb, depth, found)
        status = self.filter.update(found, self.arm.time) if found is not None else "none"
        self.frames.append(dict(t=self.arm.time, phase=phase, status=status,
                                z=None if found is None else found["lid"].copy(),
                                estimate=self.filter.estimate,
                                sigma=None if not self.filter.ready else np.sqrt(np.diag(self.filter.covariance())),
                                error=None, angle=None))
        return status

    ### visit search view until target seen
    def search(self):
        start = solve_ikine_with_limits.solve(self.kin, tool_from_camera(look_down_pose(self.mount, 0.0)),
                                              np.asarray(SEARCH_POSE, dtype=float))
        for angle in SURVEY_ANGLES:
            self.note(f"search view {degrees(angle):+.0f} deg")
            self.arm.move(start + np.array([angle, 0.0, 0.0, 0.0, 0.0, 0.0]), MOVE_TIME)  
            self.arm.wait(SETTLE)
            for _ in range(SEARCH_FRAMES):
                self.look("search")
                self.arm.wait(FRAME_PERIOD)
            if self.filter.ready:
                self.note(f"target found, lid estimate {np.round(self.filter.estimate, 3)}")
                return True
        self.note("target not found")
        return False

    ### camera twist (camera frame) 
    def joint_rates(self, T_wc, v, omega):
        q = self.arm.observer.q
        omega_w = T_wc.R @ omega  ### into the world frame
        v_tool = T_wc.R @ v + np.cross(omega_w, self.kin.fkine(q).t - T_wc.t)  ### camera velocity moved to the tool point
        J = self.kin.jacob0(q)
        qd = J.T @ np.linalg.solve(J @ J.T + DAMPING ** 2 * np.eye(6), np.r_[v_tool, omega_w])  ### damped least squares
        return qd / max(1.0, np.abs(qd).max() / MAX_JOINT_SPEED)

    ### pbvs to straight above the estimate
    def servo(self):
        self.note("pbvs approach")
        start = self.arm.time
        t_star = np.array([0.0, 0.0, STANDOFF])  ### wanted: lid straight ahead of the camera
        unseen = 0  ### frames in a row without a usable detection
        while self.arm.time - start < SERVO_TIMEOUT:
            status = self.look("servo")
            unseen = 0 if status in ("start", "accepted", "restart") else unseen + 1
            if unseen >= LOST_FRAMES:  ### moved out of view, the estimate can no longer be trusted
                self.arm.servo(np.zeros(6), FRAME_PERIOD)
                self.note("target lost")
                return "lost"
            estimate = self.filter.estimate
            T_wc = self.camera_pose()
            goal = down(estimate + np.array([0.0, 0.0, STANDOFF]), heading(estimate, self.mount))
            t = T_wc.R.T @ (estimate - T_wc.t)  ### lid in the camera frame
            e_t = t - t_star
            e_theta = rotation_vector(T_wc.R.T @ goal.R)  ### rotation still to go, camera frame
            self.frames[-1]["error"] = float(np.linalg.norm(e_t))
            self.frames[-1]["angle"] = float(np.linalg.norm(e_theta))
            if unseen == 0 and np.linalg.norm(e_t) < POSITION_TOLERANCE and np.linalg.norm(e_theta) < ANGLE_TOLERANCE:  ### converged on a fresh detection only
                self.arm.servo(np.zeros(6), FRAME_PERIOD)
                sigma = np.sqrt(np.diag(self.filter.covariance())) * 1000
                self.note(f"pbvs converged, lid estimate {np.round(estimate, 4)}, sigma {np.round(sigma, 1)} mm")
                return "converged"
            omega = GAIN * e_theta
            v = GAIN * e_t + np.cross(t, omega)  ### cross term cancels the image motion caused by turning
            v *= min(1.0, MAX_SPEED / max(np.linalg.norm(v), 1e-9))
            omega *= min(1.0, MAX_TURN / max(np.linalg.norm(omega), 1e-9))
            self.arm.servo(self.joint_rates(T_wc, v, omega), FRAME_PERIOD)  ### the lqg tracks it for one frame
        self.note("pbvs timed out")
        return "timeout"

    ### below the camera's minimum range
    def pick(self, grip):
        estimate = self.filter.estimate
        sigma = np.sqrt(np.diag(self.filter.covariance()))
        if sigma[:2].max() > PICK_SIGMA:  ### not sure enough to go in blind
            self.note(f"too unsure to pick, sigma {np.round(sigma * 1000, 1)} mm")
            return False
        yaw = heading(estimate, self.mount)
        self.note("blind descent onto the lid")
        self.arm.move_to_pose(down(estimate + np.array([0.0, 0.0, PRE_GRASP]), yaw), MOVE_TIME)
        self.arm.move_to_pose(down(estimate, yaw), 1.5)
        self.arm.wait(0.3)
        picked = grip()  
        self.note("picked" if picked else "missed the target")
        self.arm.move_to_pose(down(estimate + np.array([0.0, 0.0, LIFT]), yaw), 1.5)
        return picked

    ### carry it over basket and let go
    def place(self, release):
        table_z = -self.locator.plane[3] / self.locator.plane[2]  ### table height from the ransac plane
        height = self.filter.estimate[2] - table_z  ### estimated target height
        floor = self.basket.centre().t  ### middle of the basket floor, part of the map
        yaw = heading(floor, self.mount)
        over = self.basket.drop(clearance=height + DROP_CLEARANCE).t 
        self.note("over the basket")
        self.arm.move_to_pose(down(over, yaw), 3.0)
        self.note("lowering into the basket")
        self.arm.move_to_pose(down(floor + np.array([0.0, 0.0, height + SET_DOWN_GAP]), yaw), 2.0) 
        self.arm.wait(0.3)
        release(floor[2])
        self.note("set down")
        self.arm.move_to_pose(down(over, yaw), 1.5)  ### back up and out of the basket

    ### run everything, grip and release come from sim
    def run(self, grip, release):
        self.note("start")
        result = dict(found=False, converged=False, picked=False, attempts=0)
        for attempt in range(1 + RETRIES):
            result["attempts"] = attempt + 1
            result["found"] = self.search()
            if not result["found"]:
                break
            outcome = self.servo()
            if outcome == "lost":
                self.filter = type(self.filter)()  ### forget old position and search
                self.note("searching again")
                continue
            result["converged"] = outcome == "converged"
            break
        if result["converged"]:
            result["picked"] = self.pick(grip)
            if result["picked"]:
                self.place(release)
        self.note("home")
        self.arm.move(self.home, 3.0)
        result["estimate"] = self.filter.estimate
        result["covariance"] = self.filter.covariance()
        return result