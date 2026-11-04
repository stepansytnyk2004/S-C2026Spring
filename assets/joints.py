import numpy as np
from scipy.linalg import expm


### joint drives, reflected to the joint through the gearbox
JOINT_INERTIA = (0.40, 0.40, 0.20, 0.05, 0.05, 0.03)     
JOINT_DAMPING = (3.0, 3.0, 1.5, 0.4, 0.4, 0.3)          
TORQUE_LIMIT = (56.0, 56.0, 28.0, 12.0, 12.0, 12.0)      
TORQUE_NOISE = (0.5, 0.5, 0.3, 0.1, 0.1, 0.05)           

### encoders
ENCODER_COUNTS = 4096                                    
ENCODER_NOISE = 2e-4                                     

### simulation and control step
CONTROL_DT = 0.002                                      


### exact zero order hold discretisation, week 5 c5
def zoh(A, B, dt):
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


### the true joints: dc motor position servos (week 8 a.10), simulator side, ground truth
class joint_plant:
    def __init__(self, q0, dt=CONTROL_DT, seed=None):
        self.dt = dt
        self.x = np.column_stack([np.asarray(q0, dtype=float), np.zeros(6)])  ### true angle and velocity of each joint
        self.rng = np.random.default_rng(seed)
        self.disturbance = np.zeros(6)  ### extra load torque on each joint (n m)
        self.models = []
        for J, b in zip(JOINT_INERTIA, JOINT_DAMPING):
            A = np.array([[0.0, 1.0], [0.0, -b / J]])
            B = np.array([[0.0], [1.0 / J]])
            self.models.append(zoh(A, B, dt))

    ### one step of the true dynamics under the commanded torques
    def step(self, torque):
        torque = np.clip(torque, -np.asarray(TORQUE_LIMIT), TORQUE_LIMIT)  ### the motors cannot give more
        noise = self.rng.normal(0.0, 1.0, 6) * np.asarray(TORQUE_NOISE)
        for joint, (Ad, Bd) in enumerate(self.models):
            self.x[joint] = Ad @ self.x[joint] + Bd[:, 0] * (torque[joint] + self.disturbance[joint] + noise[joint])

    ### what the encoders report: noisy angle rounded to whole counts, velocity is not measured
    def encoders(self):
        count = 2 * np.pi / ENCODER_COUNTS
        return np.round((self.x[:, 0] + self.rng.normal(0.0, ENCODER_NOISE, 6)) / count) * count

    ### true angles, evaluation and display only
    @property
    def q(self):
        return self.x[:, 0].copy()

    ### true velocities, evaluation only
    @property
    def qd(self):
        return self.x[:, 1].copy()