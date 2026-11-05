import numpy as np

from src.control.lqr import MODEL_DAMPING, MODEL_INERTIA, discretise, servo_model


### observers model of noise 
TORQUE_NOISE_MODEL = (0.5, 0.5, 0.3, 0.1, 0.1, 0.05)  
DISTURBANCE_RATE = (10.0, 10.0, 5.0, 1.5, 1.5, 1.0)   
ENCODER_RESOLUTION = 2 * np.pi / 4096                 
ENCODER_NOISE_MODEL = 2e-4                            


### one predict + correct step of the discrete kalman filter, week 7 b.3
def kf_step(xh, P, u, y, A, B, C, Qd, Rn):
    xh_pred = A @ xh + B @ np.atleast_1d(u)          ### predict
    P_pred = A @ P @ A.T + Qd
    S = C @ P_pred @ C.T + Rn                        ### innovation covariance
    K = P_pred @ C.T @ np.linalg.inv(S)              ### kalman gain
    nu = np.atleast_1d(y) - C @ xh_pred              ### innovation
    xh_new = xh_pred + K @ nu                        ### correct
    P_new = (np.eye(len(xh)) - K @ C) @ P_pred
    return xh_new, P_new


### kalman observer for each joint
class joint_observer:
    def __init__(self, q0, dt, inertia=MODEL_INERTIA, damping=MODEL_DAMPING):
        self.models = []
        for J, b, noise, rate in zip(inertia, damping, TORQUE_NOISE_MODEL, DISTURBANCE_RATE):
            A, B, C = servo_model(J, b)
            Ad, Bd = discretise(A, B, C, dt)
            Ao = np.block([[Ad, Bd], [np.zeros((1, 2)), np.ones((1, 1))]])  ### load acts like extra torque
            Bo = np.vstack([Bd, np.zeros((1, 1))])
            Co = np.array([[1.0, 0.0, 0.0]])  ### the encoder sees the angle only
            Qd = np.zeros((3, 3))
            Qd[:2, :2] = Bd @ Bd.T * noise ** 2  
            Qd[2, 2] = rate ** 2 * dt  
            Rn = np.array([[ENCODER_RESOLUTION ** 2 / 12 + ENCODER_NOISE_MODEL ** 2]])  
            self.models.append((Ao, Bo, Co, Qd, Rn))
        q0 = np.asarray(q0, dtype=float)
        self.x = np.column_stack([q0, np.zeros(len(q0)), np.zeros(len(q0))])
        self.P = [np.diag([1e-6, 1e-4, 1.0]) for _ in q0] 

    ### one encoder reading per joint, with the torque applied over the last step
    def update(self, torque, encoders):
        for joint, (Ao, Bo, Co, Qd, Rn) in enumerate(self.models):
            self.x[joint], self.P[joint] = kf_step(self.x[joint], self.P[joint], torque[joint], encoders[joint],
                                                   Ao, Bo, Co, Qd, Rn)

    ### estimated angles
    @property
    def q(self):
        return self.x[:, 0].copy()

    ### estimated velocities
    @property
    def qd(self):
        return self.x[:, 1].copy()

    ### estimated load torques
    @property
    def d(self):
        return self.x[:, 2].copy()

    ### one standard dev for each joint
    def sigma(self):
        return np.array([np.sqrt(np.diag(P)) for P in self.P])