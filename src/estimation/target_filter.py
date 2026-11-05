import numpy as np
from scipy.stats import chi2

from src.estimation.observer import kf_step


### target filter tuning
PROCESS_NOISE = 0.001           
SYSTEMATIC = 0.002              
GATE = chi2.ppf(0.99, 3)         
PATIENCE = 3                     


### kalman filter on target lid position
class target_filter:
    def __init__(self, process_noise=PROCESS_NOISE, systematic=SYSTEMATIC, gate=GATE, patience=PATIENCE):
        self.process_noise = process_noise
        self.systematic = systematic
        self.gate = gate
        self.patience = patience
        self.x = None
        self.P = None
        self.time = None
        self.rejected = 0  ### rejected detections in a row
        self.log = []  

    ### true once the first detection has arrived
    @property
    def ready(self):
        return self.x is not None

    ### estimated lid position
    @property
    def estimate(self):
        return None if self.x is None else self.x.copy()

    ### covariance of estimate
    def covariance(self):
        return None if self.P is None else self.P + self.systematic ** 2 * np.eye(3)

    ### start again from one detection
    def start(self, detection, time):
        self.x = np.asarray(detection["lid"], dtype=float).copy()
        self.P = np.asarray(detection["R"], dtype=float).copy()
        self.time = time
        self.rejected = 0

    ### one detection: predict to its time, gate on the innovation, then correct
    def update(self, detection, time):
        if not self.ready:
            self.start(detection, time)
            self.log.append((time, "start", 0.0))
            return "start"
        Qd = self.process_noise ** 2 * max(time - self.time, 0.0) * np.eye(3)  ### random walk since last update
        R = np.asarray(detection["R"], dtype=float)
        z = np.asarray(detection["lid"], dtype=float)
        nu = z - self.x  ### innovation
        S = self.P + Qd + R
        nis = float(nu @ np.linalg.solve(S, nu))  ### normalised innovation squared
        if nis > self.gate:
            self.rejected += 1
            if self.rejected >= self.patience: 
                self.start(detection, time)
                self.log.append((time, "restart", nis))
                return "restart"
            self.log.append((time, "rejected", nis))
            return "rejected"
        I = np.eye(3)
        self.x, self.P = kf_step(self.x, self.P, np.zeros(1), z, I, np.zeros((3, 1)), I, Qd, R)  ### static model, A = C = I
        self.time = time
        self.rejected = 0
        self.log.append((time, "accepted", nis))
        return "accepted"