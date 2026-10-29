from math import pi
from spatialmath import SE3

from assets.objects import Table
from assets.ur3 import UR3
from classes.movement import move_robot


table = Table()
ur3 = UR3(base=table.top_centre(), scene=[table])  # the arm stands on the middle of the table
ur3.add_to_env()  # passive viewer, the moves below drive it

move_robot.solve(ur3, q_target=UR3.home_q, duration=2.5)  # back to home, wherever it started
move_robot.solve(ur3, ikine=SE3(0.30, 0.10, table.height + 0.25) * SE3.Rx(pi), duration=5.0)  # pre-grasp above the table, tool pointing down

ur3.close()
