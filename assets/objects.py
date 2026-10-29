import mujoco
import numpy as np

from spatialmath import SE3


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
