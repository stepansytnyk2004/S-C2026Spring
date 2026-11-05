import numpy as np


### disturbances trials can turn on
DISTURBANCES = ("none", "nudge", "noise")
NUDGE_DISTANCE = (0.03, 0.05)     ### how far the target is pushed (m)
NUDGE_DELAY = 1.5                 ### s into the pbvs approach
NOISE_SCALE = 3.0                 ### depth noise multiplier for the noisy trials


### pushes target once during approach
class nudger:
    def __init__(self, target, model, data, rng):
        self.target = target
        self.model = model
        self.data = data
        self.rng = rng
        self.task = None
        self.at = None
        self.shift = None

    ### called by the task every camera frame
    def __call__(self, time):
        if self.at is not None or self.task is None or not self.task.events:
            return
        when, what = self.task.events[-1]
        if what == "pbvs approach" and time > when + NUDGE_DELAY:
            angle = self.rng.uniform(0.0, 2 * np.pi)
            self.shift = self.rng.uniform(*NUDGE_DISTANCE) * np.array([np.cos(angle), np.sin(angle), 0.0])
            self.target.set_pose(self.model, self.data, self.target.pos + self.shift)
            self.at = time