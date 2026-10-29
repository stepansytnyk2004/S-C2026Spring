import os
import tempfile

import mujoco
import numpy as np

from math import cos, pi, sin
from spatialmath import SE3


CACHE_DIR = os.path.join(tempfile.gettempdir(), "ir_parts_mujoco")  # converted meshes

# MuJoCo cannot read the .dae files the UTS part library ships, so convert to .obj and cache it
def part_obj(name):
    from ir_support_extra_parts import part_path
    import trimesh

    os.makedirs(CACHE_DIR, exist_ok=True)
    obj = os.path.join(CACHE_DIR, name + ".obj")
    if not os.path.isfile(obj):
        mesh = trimesh.load(str(part_path(name)), force="mesh")  # flattens the scene graph
        if mesh is None or mesh.is_empty:
            raise RuntimeError(f"UTS part {name} did not load as a mesh")
        mesh.export(obj)
    return obj


class Table:
    length, width, height = 2.1, 1.4, 0.5  # metres
    thickness = 0.04  # the top slab
    leg = 0.08        # square leg, side length
    rgba = (0.55, 0.36, 0.20, 1)

    def __init__(self, name="table", pos=(0.0, 0.0, 0.0)):
        self.name = name
        self.pos = np.asarray(pos, dtype=float)  # where the table stands on the floor

    def top_centre(self):  # pose of the middle of the tabletop, where the arm is mounted
        return SE3(self.pos[0], self.pos[1], self.pos[2] + self.height)

    def add_to_spec(self, spec):  # called while the model is being built, before it compiles
        x, y, z = self.pos
        box = mujoco.mjtGeom.mjGEOM_BOX  # the spec defaults to meshes, these are primitives

        top = spec.worldbody.add_body(
            name=f"{self.name}_top",
            pos=[x, y, z + self.height - self.thickness / 2],
        )
        top.add_geom(
            type=box,
            size=[self.length / 2, self.width / 2, self.thickness / 2],
            rgba=self.rgba,
        )

        leg_height = self.height - self.thickness
        inset = self.leg / 2 + 0.02  # tuck the legs just inside the corners
        for index, (sign_x, sign_y) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            body = spec.worldbody.add_body(
                name=f"{self.name}_leg{index}",
                pos=[
                    x + sign_x * (self.length / 2 - inset),
                    y + sign_y * (self.width / 2 - inset),
                    z + leg_height / 2,
                ],
            )
            body.add_geom(
                type=box,
                size=[self.leg / 2, self.leg / 2, leg_height / 2],
                rgba=self.rgba,
            )


class Target:
    part = "CandyJar"
    rgba = (0.55, 0.78, 0.45, 1)
    reach = (0.22, 0.38)   # near and far radius of the band, metres from the arm
    half_angle = pi / 3    # swept either side of the table's +x direction

    def __init__(self, table, name="target", seed=None):
        self.table = table
        self.name = name
        self.mesh = part_obj(self.part)
        self.height, self.radius = self.measure()
        self.rng = np.random.default_rng(seed)
        self.pos = self.sample()

    def measure(self):  # how tall and how wide the jar is, read off the mesh itself
        import trimesh

        bounds = trimesh.load(self.mesh, force="mesh").bounds
        extents = bounds[1] - bounds[0]
        return float(extents[2]), float(max(extents[0], extents[1]) / 2)

    def sample(self):  # a fresh position in the band, the randomisation between trials
        mount = self.table.top_centre().t
        radius = self.rng.uniform(*self.reach)
        angle = self.rng.uniform(-self.half_angle, self.half_angle)
        return np.array([
            mount[0] + radius * cos(angle),
            mount[1] + radius * sin(angle),
            mount[2],  # the mesh origin is the base of the jar, so it stands on the tabletop
        ])

    def pose(self):  # where the jar actually is, the ground truth sensing would estimate
        return SE3(self.pos[0], self.pos[1], self.pos[2])

    def grasp(self, clearance=0.0):  # tool pointing straight down over the lid
        return SE3(self.pos[0], self.pos[1], self.pos[2] + self.height + clearance) * SE3.Rx(pi)

    def add_to_spec(self, spec):  # called while the model is being built, before it compiles
        spec.add_mesh(name=self.name, file=self.mesh)  # register the file once
        body = spec.worldbody.add_body(name=self.name, pos=self.pos.tolist())
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=self.name,
            rgba=self.rgba,
        )


class Basket:
    half = 0.12    # interior half-width
    height = 0.10
    wall = 0.01
    rgba = (0.30, 0.45, 0.75, 1)
    offset = (-0.30, 0.0)  # fixed spot on the table, measured from the arm

    def __init__(self, table, name="basket"):
        self.table = table
        self.name = name
        mount = table.top_centre().t
        self.pos = np.array([mount[0] + self.offset[0], mount[1] + self.offset[1], mount[2]])

    def centre(self):  # middle of the basket floor
        return SE3(self.pos[0], self.pos[1], self.pos[2] + self.wall)

    def drop(self, clearance=0.05):  # tool pointing down over the opening
        return SE3(self.pos[0], self.pos[1], self.pos[2] + self.height + clearance) * SE3.Rx(pi)

    def add_to_spec(self, spec):  # called while the model is being built, before it compiles
        x, y, z = self.pos
        box = mujoco.mjtGeom.mjGEOM_BOX

        floor = spec.worldbody.add_body(name=f"{self.name}_floor", pos=[x, y, z + self.wall / 2])
        floor.add_geom(
            type=box,
            size=[self.half, self.half, self.wall / 2],
            rgba=self.rgba,
        )

        # two walls across x, two across y
        for index, (sign_x, sign_y) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
            size = (
                [self.wall / 2, self.half, self.height / 2]
                if sign_x
                else [self.half, self.wall / 2, self.height / 2]
            )
            body = spec.worldbody.add_body(
                name=f"{self.name}_wall{index}",
                pos=[
                    x + sign_x * self.half,
                    y + sign_y * self.half,
                    z + self.height / 2,
                ],
            )
            body.add_geom(type=box, size=size, rgba=self.rgba)
