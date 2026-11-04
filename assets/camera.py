import mujoco
import numpy as np
from math import cos, pi, radians, sin, tan
from spatialmath import SE3


### camera constraints set up
CAMERA_NAME = "wrist_cam"                  
CAMERA_FOVY = 42.0                        
CAMERA_SIZE = (640, 480)                  
CAMERA_POS = (0.06, 0.0, -0.02)           
DEPTH_RANGE = (0.17, 3.0)            

### search views over the jar band
SURVEY_ANGLES = (-0.7, 0.0, 0.7)          
SURVEY_RADIUS = 0.30                      
SURVEY_HEIGHT = 0.40                      

### depth noise model
DEPTH_NOISE = (0.0015, 0.0022)            
DEPTH_DROPOUT = 0.01                      
DEPTH_EDGE = 0.02                         


### adds camera and visual block as housing
def add_camera(tool):
    tool.add_camera(name=CAMERA_NAME, pos=list(CAMERA_POS), quat=[0, 1, 0, 0], fovy=CAMERA_FOVY)  ### half turn about x, looks along the tool's +z
    box = mujoco.mjtGeom.mjGEOM_BOX
    tool.add_geom(type=box, size=[0.0125, 0.045, 0.0125], ### adds camera box
                  pos=[CAMERA_POS[0], CAMERA_POS[1], CAMERA_POS[2] - 0.0125], rgba=[0.15, 0.15, 0.17, 1])
    tool.add_geom(type=box, size=[CAMERA_POS[0] / 2, 0.015, 0.003], ### adds camera connector
                  pos=[CAMERA_POS[0] / 2, CAMERA_POS[1], CAMERA_POS[2] - 0.0125], rgba=[0.7, 0.7, 0.7, 1])


### end effector picking up the jar (no gripper atm)
class holder:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool0")  
        self.carrying = None  ### target being held
        self.offset = None    ### targets pose

    ### tool0 pose
    def tool_pose(self):
        return self.data.xpos[self.tool].copy(), self.data.xmat[self.tool].reshape(3, 3).copy()

    ### pick up, target maintains orientation
    def attach(self, target):
        pos, rot = self.tool_pose()
        rot_jar = np.zeros(9)
        mujoco.mju_quat2Mat(rot_jar, target.quat)  
        self.offset = (rot.T @ (target.pos - pos), rot.T @ rot_jar.reshape(3, 3))  
        self.carrying = target

    ### move held jar with tool
    def carry(self):
        pos, rot = self.tool_pose()
        pos_offset, rot_offset = self.offset
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rot @ rot_offset).flatten()) 
        self.carrying.set_pose(self.model, self.data, pos + rot @ pos_offset, quat)

    ### target release
    def release(self, floor=None):
        target = self.carrying
        self.carrying = None
        self.offset = None
        if target is not None and floor is not None:
            pos = target.pos.copy()
            pos[2] = floor  ### straight down to the floor
            target.set_pose(self.model, self.data, pos, [1.0, 0.0, 0.0, 0.0])
        return target


### the wrist d435: colour and depth images from wrist_cam
class rgbd:
    def __init__(self, model, data, name=CAMERA_NAME, size=CAMERA_SIZE, seed=None):
        self.model = model
        self.data = data
        self.width, self.height = size
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)  ### offscreen renderer for camera
        self.rng = np.random.default_rng(seed)  ### random numbers for depth noise
        self.noise_scale = 1.0  ### multiplies depth noise

        ### true intrinsics from the model, evaluation only, the pipeline uses the calibrated K
        f = (self.height / 2) / tan(radians(model.cam_fovy[self.id]) / 2)
        self.K = np.array([[f, 0.0, (self.width - 1) / 2],
                           [0.0, f, (self.height - 1) / 2],
                           [0.0, 0.0, 1.0]])

    ### one frame: rgb (h x w x 3) and noisy depth (h x w, metres)
    def read(self, noise=True):
        self.renderer.update_scene(self.data, camera=self.id)
        rgb = self.renderer.render().copy()  ### colour image

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.id)
        exact = self.renderer.render().copy()  ### perfect depth along optical axis
        self.renderer.disable_depth_rendering()

        depth = exact.copy()
        if noise:
            sigma = self.noise_scale * (DEPTH_NOISE[0] + DEPTH_NOISE[1] * exact ** 2)  ### noise grows with range squared
            depth = exact + self.rng.normal(0.0, 1.0, exact.shape) * sigma
            jump = np.abs(np.gradient(exact, axis=0)) + np.abs(np.gradient(exact, axis=1))  
            depth[(jump > DEPTH_EDGE) | (self.rng.random(exact.shape) < DEPTH_DROPOUT)] = 0.0 
        depth[(exact < DEPTH_RANGE[0]) | (exact > DEPTH_RANGE[1])] = 0.0  ### too close or too far, no return
        return rgb, depth

    ### true camera pose in world
    def pose(self):
        rot = self.data.cam_xmat[self.id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])  
        return SE3.Rt(rot, self.data.cam_xpos[self.id].copy(), check=False)  ### evaluation only

    ### world points into this camera's image with true pose and k
    def project(self, points):
        return project(self.K, self.pose(), points)  ### evaluation only

    def close(self):
        self.renderer.close()


### camera pose over target looking straight down
def look_down_pose(mount, angle, radius=SURVEY_RADIUS, height=SURVEY_HEIGHT):
    centre = SE3(mount[0] + radius * cos(angle), mount[1] + radius * sin(angle), mount[2] + height)
    return centre * SE3.Rz(angle + pi / 2) * SE3.Rx(pi)  ### turn to arc, then flip to look down

### for the calibration views
def look_at_from(centre, distance, tilt=0.0, azim=0.0, roll=0.0):
    return SE3(*centre) * SE3.Rz(azim) * SE3.Rx(tilt) * SE3.Tz(distance) * SE3.Rx(pi) * SE3.Rz(azim + roll)  ### lean, move out, look back, undo the azimuth

### where tool0 must be for the camera to sit at camera_pose, using design mount
def tool_from_camera(camera_pose):
    return camera_pose * SE3(*CAMERA_POS).inv()  


### world points (n x 3) -> pixels (n x 2) and depth
def project(K, camera_pose, points):
    p = (np.atleast_2d(points) - camera_pose.t) @ camera_pose.R  ### each row into the camera frame
    uv = (p @ K.T)[:, :2] / p[:, 2:3]  ### pinhole projection, divide by depth
    return uv, p[:, 2]