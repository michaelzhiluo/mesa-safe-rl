import numpy as np


def autograsp_dynamics(prev_target_qpos, action, gripper_closed, gripper_zpos, zthresh, reopen, grasp_condition):
    target_qpos = np.zeros_like(prev_target_qpos)
    target_qpos[:4] = action[:4] + prev_target_qpos[:4]

    if gripper_zpos < zthresh:
        gripper_closed = True
    elif reopen and not grasp_condition:
        gripper_closed = False

    if gripper_closed:
        target_qpos[4] = 1
    else:
        target_qpos[4] = -1

    return target_qpos, gripper_closed

def no_rot_dynamics(prev_target_qpos, action, gripper_closed, gripper_zpos, zthresh, reopen, grasp_condition):
    target_qpos = np.zeros_like(prev_target_qpos)
    target_qpos[:3] = action[:3] + prev_target_qpos[:3]
    target_qpos[4] = action[3]
    return target_qpos, gripper_closed

def clip_target_qpos(target, lb, ub):
    target[:len(lb)] = np.clip(target[:len(lb)], lb, ub)
    return target
