import cv2
import numpy as np
import open3d as o3d

from spatialmath import SE3


### cloud cleaning, week 4 part 3
VOXEL = 0.005                     ### voxel leaf size (m)
NEIGHBOURS = 20                   ### outlier removal neighbours
STD_RATIO = 2.0                   ### outlier removal threshold

### plane and objects, week 4 part 4
PLANE_THRESHOLD = 0.01            ### ransac inlier distance (m)
ABOVE = 0.01                      ### points this far above the plane are objects (m)
CLUSTER_EPS = 0.02                ### dbscan neighbour distance (m)
CLUSTER_MIN = 30                  ### dbscan smallest cluster

### what counts as the jar, from its known size
JAR_WIDTH = (0.04, 0.12)          ### width across (m)
JAR_HEIGHT = (0.03, 0.15)         ### height above the table (m)

### measurement uncertainty
SENSOR_NOISE = (0.0015, 0.0022)   ### d435 depth noise, sigma = a + b * d^2
CALIBRATION_FLOOR = 0.003         ### calibration error that does not average out (m)


### pixel rays from the calibrated K and distortion, worked out once
class depth_projector:
    def __init__(self, K, dist, size, stride=2):
        width, height = size
        u, v = np.meshgrid(np.arange(0, width, stride), np.arange(0, height, stride))  ### every 2nd pixel
        self.u, self.v = u.ravel(), v.ravel()
        pixels = np.column_stack([self.u, self.v]).astype(np.float64).reshape(-1, 1, 2)
        rays = cv2.undistortPoints(pixels, np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64))  ### normalised, undistorted
        self.rays = np.column_stack([rays.reshape(-1, 2), np.ones(len(self.u))])

    ### depth image -> points in the camera frame, p = d K^-1 [u v 1]
    def points(self, depth):
        d = depth[self.v, self.u]
        valid = d > 0  ### 0 = no return
        return self.rays[valid] * d[valid, None]


### week 4 find_objects: clean, crop, ransac plane, points above it, dbscan, keep the jar sized cluster
def find_jar(points, roi, plane=None):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(VOXEL)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=NEIGHBOURS, std_ratio=STD_RATIO)
    cloud = cloud.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=roi[0], max_bound=roi[1]))  ### workspace only
    if len(cloud.points) < CLUSTER_MIN:
        return None, plane

    if plane is None:  ### table plane found once, then kept
        model, _ = cloud.segment_plane(PLANE_THRESHOLD, 3, 500)
        plane = np.asarray(model)
        if plane[2] < 0:  ### normal pointing up
            plane = -plane
    table_z = -plane[3] / plane[2]

    pts = np.asarray(cloud.points)
    above = pts[pts @ plane[:3] + plane[3] > ABOVE]  ### signed distance to the plane
    if len(above) < CLUSTER_MIN:
        return None, plane
    labels = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(above)).cluster_dbscan(CLUSTER_EPS, CLUSTER_MIN))

    best = None
    for label in range(labels.max() + 1 if len(labels) else 0):
        cluster = above[labels == label]
        lower, upper = cluster.min(axis=0), cluster.max(axis=0)
        width = max(upper[0] - lower[0], upper[1] - lower[1])
        lid_z = np.percentile(cluster[:, 2], 95)  ### lid height, robust to noisy points
        height = lid_z - table_z
        if not (JAR_WIDTH[0] <= width <= JAR_WIDTH[1] and JAR_HEIGHT[0] <= height <= JAR_HEIGHT[1]):  ### not jar sized
            continue
        if best is None or len(cluster) > best["count"]:
            best = dict(lid=np.array([(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, lid_z]),  ### middle of the footprint
                        height=float(height), width=float(width), count=len(cluster), cluster=cluster)
    return best, plane


### measurement covariance R for the jar filter
def measurement_covariance(detection, distance):
    sigma_depth = SENSOR_NOISE[0] + SENSOR_NOISE[1] * distance ** 2
    top = int(np.sum(detection["cluster"][:, 2] > detection["lid"][2] - 0.01))  ### points on the lid
    sigma_xy = np.hypot(CALIBRATION_FLOOR, VOXEL / np.sqrt(24))  ### calibration and voxel quantisation
    sigma_z = np.hypot(CALIBRATION_FLOOR, sigma_depth / np.sqrt(max(top, 1)))  ### calibration and averaged depth noise
    return np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2])


### one depth frame and the tool pose -> jar lid measurement and its R, or none
class jar_locator:
    def __init__(self, calibration, roi):
        self.K = np.asarray(calibration["K"], dtype=float)
        self.dist = np.asarray(calibration["dist"], dtype=float)
        self.T_tool_cam = SE3(np.asarray(calibration["T_tool_cam"], dtype=float), check=False)  ### hand-eye result
        self.projector = depth_projector(self.K, self.dist, tuple(int(s) for s in calibration["size"]))
        self.roi = (np.asarray(roi[0], dtype=float), np.asarray(roi[1], dtype=float))
        self.plane = None

    def locate(self, depth, T_world_tool):
        T_world_cam = T_world_tool * self.T_tool_cam  ### arm fk then the calibrated camera mount
        points = self.projector.points(depth)
        if len(points) < CLUSTER_MIN:
            return None
        world = points @ T_world_cam.R.T + T_world_cam.t  ### camera frame -> world frame
        detection, self.plane = find_jar(world, self.roi, self.plane)
        if detection is None:
            return None
        detection["R"] = measurement_covariance(detection, np.linalg.norm(detection["lid"] - T_world_cam.t))
        detection["camera"] = T_world_cam
        return detection