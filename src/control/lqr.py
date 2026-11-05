import control as ct
import numpy as np


### controller's model of joint drives
MODEL_INERTIA = (0.40, 0.40, 0.20, 0.05, 0.05, 0.03)     
MODEL_DAMPING = (3.0, 3.0, 1.5, 0.4, 0.4, 0.3)           
MODEL_TORQUE_LIMIT = (56.0, 56.0, 28.0, 12.0, 12.0, 12.0) 

### largest error in each state
POSITION_ERROR = 0.002     
VELOCITY_ERROR = 0.05      
INTEGRAL_ERROR = 0.002     


### one joint as dc motor position servo
def servo_model(inertia, damping):
    A = np.array([[0.0, 1.0], [0.0, -damping / inertia]])
    B = np.array([[0.0], [1.0 / inertia]])
    C = np.array([[1.0, 0.0]])
    return A, B, C


### zero order hold model 
def discretise(A, B, C, dt):
    model = ct.ss(A, B, C, np.zeros((1, 1))).sample(dt, method="zoh")
    return np.asarray(model.A), np.asarray(model.B)


### lqr with integral action on augmented system
def lqi_gain(Ad, Bd, C, dt, torque_limit):
    n = Ad.shape[0]
    Aa = np.block([[Ad, np.zeros((n, 1))], [-dt * C, np.ones((1, 1))]])
    Ba = np.vstack([Bd, np.zeros((1, 1))])
    Q = np.diag([1 / POSITION_ERROR ** 2, 1 / VELOCITY_ERROR ** 2, 1 / INTEGRAL_ERROR ** 2])
    R = np.array([[1 / torque_limit ** 2]])  ### full torque most accepted
    K, _, poles = ct.dlqr(Aa, Ba, Q, R)
    return K[:, :n], K[:, n:], poles


### joint torques from the estimated state
class joint_controller:
    def __init__(self, dt, inertia=MODEL_INERTIA, damping=MODEL_DAMPING, torque_limit=MODEL_TORQUE_LIMIT):
        self.inertia = np.asarray(inertia, dtype=float)
        self.damping = np.asarray(damping, dtype=float)
        self.limit = np.asarray(torque_limit, dtype=float)
        self.dt = dt
        self.K, self.Ki, self.poles = [], [], []
        for J, b, limit in zip(self.inertia, self.damping, self.limit):
            A, B, C = servo_model(J, b)
            Ad, Bd = discretise(A, B, C, dt)
            K, Ki, poles = lqi_gain(Ad, Bd, C, dt, limit)
            self.K.append(K.ravel())
            self.Ki.append(float(Ki.ravel()[0]))
            self.poles.append(poles)
        self.integral = np.zeros(len(self.inertia))  ### accumulated position error of each joint

    def torque(self, q_hat, qd_hat, d_hat, q_ref, qd_ref, qdd_ref):
        feedforward = self.inertia * qdd_ref + self.damping * qd_ref - d_hat  ### what the reference less the load
        torque = np.zeros(len(self.inertia))
        for joint in range(len(self.inertia)):
            error = np.array([q_hat[joint] - q_ref[joint], qd_hat[joint] - qd_ref[joint]])
            wanted = feedforward[joint] - self.K[joint] @ error - self.Ki[joint] * self.integral[joint]
            torque[joint] = np.clip(wanted, -self.limit[joint], self.limit[joint])
            if torque[joint] == wanted:  ### anti-windup no integrating while saturated
                self.integral[joint] += self.dt * (q_ref[joint] - q_hat[joint])
        return torque