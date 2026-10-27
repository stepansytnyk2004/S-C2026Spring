import os
import shutil
import tempfile
import mujoco

import numpy as np

from math import pi
from spatialmath import SE3


MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")  # meshes beside this file
CACHE_DIR = os.path.join(tempfile.gettempdir(), "ur3_mujoco")  # converted meshes


class UR3:

    home_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def __init__(self, base=None):
        self.links = [
            ("base_link",      (0, 0, 0),              (0, 0, 0),       None, None,        "base"),
            ("shoulder_link",  (0, 0, 0.1519),         (0, 0, 0),       "z",  (-360, 360), "shoulder"),
            ("upper_arm_link", (0, 0.1198, 0),         (0, pi / 2, 0),  "y",  (-360, 360), "upperarm"),
            ("forearm_link",   (0, -0.0925, 0.24365),  (0, 0, 0),       "y",  (-360, 360), "forearm"),
            ("wrist_1_link",   (0, 0, 0.21325),        (0, pi / 2, 0),  "y",  (-360, 360), "wrist1"),
            ("wrist_2_link",   (0, 0.08505, 0),        (0, 0, 0),       "z",  (-360, 360), "wrist2"),
            ("wrist_3_link",   (0, 0, 0.08535),        (0, 0, 0),       "y",  (-360, 360), "wrist3"),
            ("tool0",          (0, 0.0819, 0),         (-pi / 2, 0, 0), None, None,        None),
        ]

        self.name = "UR3"
        self.joint_names = [link[0] for link in self.links if link[3]]  # links that carry a joint
        self.model, self.data = self.add_to_mujoco()
        self.viewer = None

        # qpos index of each joint, and the body the base pose is written to
        self._qadr = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self.joint_names]
        self._base_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        self.base = SE3() if base is None else base  # identity sits it at the origin
        self.q = list(self.home_q)  # copied, so the class attribute is not edited
        self.update()

    def qlim(self, lower, upper):  # degrees -> radians
        return [lower * pi / 180, upper * pi / 180]

    # build the model from the link table, returns (MjModel, MjData)
    def add_to_mujoco(self):
        spec = mujoco.MjSpec()
        spec.modelname = self.name
        spec.compiler.degree = False  # joint ranges below are in radians
        spec.compiler.balanceinertia = True  # mesh inertias can be slightly off
        spec.option.gravity = [0, 0, 0]  # display model, it holds its pose

        spec.default.geom.type = mujoco.mjtGeom.mjGEOM_MESH
        spec.default.geom.rgba = [0.7, 0.7, 0.7, 1]
        spec.default.geom.density = 1400  # mesh volume x density gives about 11 kg
        spec.default.geom.contype = 0
        spec.default.geom.conaffinity = 0

        spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

        meshes = mesh_files([link[5] for link in self.links if link[5]])
        parent = spec.worldbody  # the first link hangs off the world

        # nest each link inside the one before it
        for name, pos, euler, axis, qlim, mesh in self.links:
            body = parent.add_body(name=name, pos=pos, quat=euler2quat(euler))
            if axis:
                unit = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]  # axis letter -> vector
                body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=unit, range=self.qlim(*qlim))
            if mesh:
                spec.add_mesh(name=mesh, file=meshes[mesh])  # register the file once
                body.add_geom(meshname=mesh)  # then draw it on this link
            else:   # frame-only link, a token mass keeps the compiler happy
                body.mass = 1e-4
                body.inertia = [1e-6, 1e-6, 1e-6]
                body.explicitinertial = True
            parent = body  # the next link nests in this one

        self.spec = spec
        model = spec.compile()
        return model, mujoco.MjData(model)

    # write the current base pose and joint angles into the model
    def update(self):
        T = self.base.A if isinstance(self.base, SE3) else np.asarray(self.base, dtype=float)  # SE3 or 4x4
        self.model.body_pos[self._base_body] = T[0:3, 3]  # translation
        self.model.body_quat[self._base_body] = mat2quat(T[0:3, 0:3])  # rotation

        self.data.qpos[self._qadr] = np.asarray(self.q, dtype=float).reshape(-1)
        mujoco.mj_forward(self.model, self.data)  # recompute link poses
        self.sync()

    # open the viewer, imported here because it needs a window
    def add_to_env(self, show_ui=False, block=False):   # block=True hands the loop to MuJoCo until the window closes
        import mujoco.viewer

        if block:
            return mujoco.viewer.launch(self.model, self.data)

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=show_ui, show_right_ui=show_ui)
        self.sync()
        return self.viewer

    def sync(self):  # redraw, if a viewer is open
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def __repr__(self):
        return f"<{self.name} q={np.round(self.q, 3).tolist()}>"


def euler2quat(euler):  # xyz euler angles -> MuJoCo (w, x, y, z) quaternion
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.asarray(euler, dtype=float), "xyz")
    return quat


def mat2quat(rot):  # rotation matrix -> MuJoCo (w, x, y, z) quaternion
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=float).flatten())  # wants it flat
    return quat


# find a MuJoCo-readable file per mesh, returns {name: path}
def mesh_files(names):
    import rtbdata

    src = os.path.join(os.path.dirname(rtbdata.__file__), "xacro", "ur_description", "meshes", "ur3")
    os.makedirs(CACHE_DIR, exist_ok=True)
    found = {}

    for name in names:
        local = [os.path.join(MESH_DIR, name + ext) for ext in (".obj", ".stl")]
        local = [path for path in local if os.path.isfile(path)]
        if local:
            found[name] = local[0]  # prefer the .obj
            continue

        # MuJoCo cannot read .dae, so convert it once and cache the .obj
        obj = os.path.join(CACHE_DIR, name + ".obj")
        if os.path.isfile(obj) or dae_to_obj(os.path.join(src, "visual", name + ".dae"), obj):
            found[name] = obj
            continue

        # fallback, the coarser collision mesh is already .stl
        stl = os.path.join(CACHE_DIR, name + ".stl")
        if not os.path.isfile(stl):
            shutil.copyfile(os.path.join(src, "collision", name + ".stl"), stl)
        found[name] = stl

    return found


def dae_to_obj(src, dst):  # needs trimesh + pycollada, False if unavailable
    try:
        import trimesh

        mesh = trimesh.load(src, force="mesh")
        if mesh is None or mesh.is_empty:
            return False
        mesh.export(dst)
        return True
    except Exception:
        return False  # caller falls back to the .stl
