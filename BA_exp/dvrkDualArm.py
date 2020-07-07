from .dvrkArm import dvrkArm
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
import BA_exp.utils as U

class dvrkDualArm(object):
    def __init__(self):
        self.arm1 = dvrkArm('/PSM1')
        self.arm2 = dvrkArm('/PSM2')

    def set_pose(self, pos1=[], rot1=[], pos2=[], rot2=[], wait_callback=True):
        msg1 = Pose()
        msg2 = Pose()
        if not rot1==[]:
            rot_transformed1 = self.arm1._dvrkArm__set_rot_transform(rot1)
        if not rot2==[]:
            rot_transformed2 = self.arm2._dvrkArm__set_rot_transform(rot2)

        if pos1==[]:
            msg1.position.x = self.arm1._dvrkArm__act_pos[0]
            msg1.position.y = self.arm1._dvrkArm__act_pos[1]
            msg1.position.z = self.arm1._dvrkArm__act_pos[2]
        else:
            msg1.position.x = pos1[0]
            msg1.position.y = pos1[1]
            msg1.position.z = pos1[2]

        if rot1==[]:
            msg1.orientation.x = self.arm1._dvrkArm__act_rot[0]
            msg1.orientation.y = self.arm1._dvrkArm__act_rot[1]
            msg1.orientation.z = self.arm1._dvrkArm__act_rot[2]
            msg1.orientation.w = self.arm1._dvrkArm__act_rot[3]
        else:
            msg1.orientation.x = rot_transformed1[0]
            msg1.orientation.y = rot_transformed1[1]
            msg1.orientation.z = rot_transformed1[2]
            msg1.orientation.w = rot_transformed1[3]

        if pos2 == []:
            msg2.position.x = self.arm2._dvrkArm__act_pos[0]
            msg2.position.y = self.arm2._dvrkArm__act_pos[1]
            msg2.position.z = self.arm2._dvrkArm__act_pos[2]
        else:
            msg2.position.x = pos2[0]
            msg2.position.y = pos2[1]
            msg2.position.z = pos2[2]

        if rot2 == []:
            msg2.orientation.x = self.arm2._dvrkArm__act_rot[0]
            msg2.orientation.y = self.arm2._dvrkArm__act_rot[1]
            msg2.orientation.z = self.arm2._dvrkArm__act_rot[2]
            msg2.orientation.w = self.arm2._dvrkArm__act_rot[3]
        else:
            msg2.orientation.x = rot_transformed2[0]
            msg2.orientation.y = rot_transformed2[1]
            msg2.orientation.z = rot_transformed2[2]
            msg2.orientation.w = rot_transformed2[3]

        if wait_callback:
            self.arm1._dvrkArm__goal_reached_event.clear()
            self.arm2._dvrkArm__goal_reached_event.clear()
            self.arm1._dvrkArm__set_position_goal_cartesian_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_cartesian_pub.publish(msg2)
            return self.arm1._dvrkArm__goal_reached_event.wait(10) and self.arm2._dvrkArm__goal_reached_event.wait(10)  # 10 seconds at most:
        else:
            self.arm1._dvrkArm__set_position_goal_cartesian_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_cartesian_pub.publish(msg2)
            return True

    def set_jaw(self, jaw1=[], jaw2=[], wait_callback=True):
        msg1 = JointState()
        msg2 = JointState()
        if jaw1==[]:
            msg1.position = self.arm1._dvrkArm__act_jaw
        else:
            msg1.position = jaw1

        if jaw2==[]:
            msg2.position = self.arm2._dvrkArm__act_jaw
        else:
            msg2.position = jaw2

        if wait_callback:
            self.arm1._dvrkArm__goal_reached_event.clear()
            self.arm2._dvrkArm__goal_reached_event.clear()
            self.arm1._dvrkArm__set_position_goal_jaw_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_jaw_pub.publish(msg2)
            return self.arm1._dvrkArm__goal_reached_event.wait(10) and self.arm2._dvrkArm__goal_reached_event.wait(10)  # 10 seconds at most
        else:
            self.arm1._dvrkArm__set_position_goal_jaw_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_jaw_pub.publish(msg2)
            return True

    def set_joint(self, joint1=[], joint2=[], wait_callback=True):
        msg1 = JointState()
        msg2 = JointState()
        if joint1==[]:
            msg1.position = self.arm1._dvrkArm__act_joint
        else:
            msg1.position = joint1

        if joint2==[]:
            msg2.position = self.arm2._dvrkArm__act_joint
        else:
            msg2.position = joint2
        if wait_callback:
            self.arm1._dvrkArm__goal_reached_event.clear()
            self.arm2._dvrkArm__goal_reached_event.clear()
            self.arm1._dvrkArm__set_position_goal_joint_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_joint_pub.publish(msg2)
            return self.arm1._dvrkArm__goal_reached_event.wait(10) and self.arm2._dvrkArm__goal_reached_event.wait(10)    # 10 seconds at most
        else:
            self.arm1._dvrkArm__set_position_goal_joint_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_joint_pub.publish(msg2)
            return True

    def set_arm_position(self, pos1=[], pos2=[], wait_callback=True):
        msg1 = JointState()
        msg2 = JointState()
        if pos1==[]:
            msg1.position = self.arm1._dvrkArm__act_joint
        else:
            j1, j2, j3 = self.arm1.inverse_kin_arm(pos1)
            msg1.position = [j1, j2, j3, 0.0, 0.0, 0.0]

        if pos2==[]:
            msg2.position = self.arm2._dvrkArm__act_joint
        else:
            j1, j2, j3 = self.arm2.inverse_kin_arm(pos2)
            msg2.position = [j1, j2, j3, 0.0, 0.0, 0.0]

        if wait_callback:
            self.arm1._dvrkArm__goal_reached_event.clear()
            self.arm2._dvrkArm__goal_reached_event.clear()
            self.arm1._dvrkArm__set_position_goal_joint_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_joint_pub.publish(msg2)
            return self.arm1._dvrkArm__goal_reached_event.wait(10) and self.arm2._dvrkArm__goal_reached_event.wait(10)    # 10 seconds at most
        else:
            self.arm1._dvrkArm__set_position_goal_joint_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_joint_pub.publish(msg2)
            return True

if __name__ == "__main__":
    p = dvrkDualArm()
    import numpy as np
    while True:
        pos11 = [-0.1, -0.1, -0.14]
        rot11 = np.array([0, -60, -60]) * np.pi / 180.  # ZYX Euler angle in (deg)
        q11 = U.euler_to_quaternion(rot11)
        jaw11 = [0]
        pos12 = [-0.05, -0.05, -0.14]
        rot12 = np.array([0, -60, -60]) * np.pi / 180.  # ZYX Euler angle in (deg)
        q12 = U.euler_to_quaternion(rot12)
        jaw12 = [0]
        p.set_pose(pos1=pos11,rot1=q11,pos2=pos12,rot2=q12)
        # p.set_pose(pos1=pos11, rot1=q11)
        p.set_jaw(jaw1=jaw11, jaw2=jaw12)
        # p.set_jaw(jaw2=jaw12)

        pos21 = [0.1, 0.1, -0.14]
        rot21 = np.array([0, 60, 60]) * np.pi / 180.  # ZYX Euler angle in (deg)
        q21 = U.euler_to_quaternion(rot21)
        jaw21 = [50*np.pi/180]
        pos22 = [0.05, 0.05, -0.14]
        rot22 = np.array([0, 60, 60]) * np.pi / 180.  # ZYX Euler angle in (deg)
        q22 = U.euler_to_quaternion(rot22)
        jaw22 = [50*np.pi/180]
        jaw22 = [50*np.pi/180]
        p.set_pose(pos1=pos21, rot1=q21, pos2=pos22, rot2=q22)
        # p.set_pose(pos1=pos21, rot1=q21)
        p.set_jaw(jaw1=jaw21, jaw2=jaw22)
        # p.set_jaw(jaw2=jaw22)