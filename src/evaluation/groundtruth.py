import numpy as np


### only reads of simulator's true state outside simulator itself: used to score trials
### true lid centre of the target
def true_lid(target):
    return target.pos + np.array([0.0, 0.0, target.height])


### tool position minus the true lid (mm)
def grip_error(holder, target):
    pos, _ = holder.tool_pose()
    return (pos - true_lid(target)) * 1000


### whether the target is standing inside the basket
def in_basket(target, basket):
    return bool(np.all(np.abs(target.pos[:2] - basket.pos[:2]) < basket.half - target.radius / 2))


### normalised estimation error 
def target_nees(estimate, covariance, target):
    error = np.asarray(estimate) - true_lid(target)
    return float(error @ np.linalg.solve(covariance, error))


### joint observer accuracy over one trial from arms log
def joint_observer_stats(history):
    if len(history["t"]) == 0:
        return float("nan"), float("nan")
    velocity_error = history["qd_true"] - history["qd_hat"]
    state_error = np.stack([history["q_true"] - history["q_hat"], velocity_error], axis=2)
    P = history["P"][:, :, :2, :2]
    nees = np.einsum("kji,kjil,kjl->kj", state_error, np.linalg.inv(P), state_error)
    return float(np.sqrt((velocity_error ** 2).mean())), float(nees.mean())