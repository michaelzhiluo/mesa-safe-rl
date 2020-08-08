import numpy as np
import threading
import PyKDL
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf_conversions import posemath
import utils as U


class dvrkArm(object):
    """Simple arm API wrapping around ROS messages
    """

    def __init__(self, arm_name, ros_namespace='/dvrk'):

        # continuous publish from dvrk_bridge
        # actual(current) values
        self.__act_pose_frame = PyKDL.Frame()
        self.__act_pos = []
        self.__act_rot = []  # quaternion
        self.__act_jaw = []
        self.__act_joint = []

        # data members, event based
        self.__arm_name = arm_name
        self.__ros_namespace = ros_namespace
        self.__goal_reached_event = threading.Event()
        self.__get_position_event = threading.Event()
        self.__get_joint_event = threading.Event()
        self.__get_jaw_event = threading.Event()

        self.__sub_list = []
        self.__pub_list = []

        # publisher
        self.__full_ros_namespace = self.__ros_namespace + self.__arm_name
        self.__set_position_joint_pub = rospy.Publisher(
            self.__full_ros_namespace + '/set_position_joint',
            JointState,
            latch=True,
            queue_size=1)
        self.__set_position_goal_joint_pub = rospy.Publisher(
            self.__full_ros_namespace + '/set_position_goal_joint',
            JointState,
            latch=True,
            queue_size=1)
        self.__set_position_cartesian_pub = rospy.Publisher(
            self.__full_ros_namespace + '/set_position_cartesian',
            Pose,
            latch=True,
            queue_size=1)
        self.__set_position_goal_cartesian_pub = rospy.Publisher(
            self.__full_ros_namespace + '/set_position_goal_cartesian',
            Pose,
            latch=True,
            queue_size=1)
        self.__set_position_jaw_pub = rospy.Publisher(
            self.__full_ros_namespace + '/set_position_jaw',
            JointState,
            latch=True,
            queue_size=1)
        self.__set_position_goal_jaw_pub = rospy.Publisher(
            self.__full_ros_namespace + '/set_position_goal_jaw',
            JointState,
            latch=True,
            queue_size=1)

        self.__pub_list = [
            self.__set_position_joint_pub, self.__set_position_goal_joint_pub,
            self.__set_position_cartesian_pub,
            self.__set_position_goal_cartesian_pub,
            self.__set_position_jaw_pub, self.__set_position_goal_jaw_pub
        ]

        self.__sub_list = [
            rospy.Subscriber(self.__full_ros_namespace + '/goal_reached', Bool,
                             self.__goal_reached_cb),
            rospy.Subscriber(
                self.__full_ros_namespace + '/position_cartesian_current',
                PoseStamped, self.__position_cartesian_current_cb),
            rospy.Subscriber(
                self.__full_ros_namespace + '/state_joint_current', JointState,
                self.__position_joint_current_cb),
            rospy.Subscriber(self.__full_ros_namespace + '/state_jaw_current',
                             JointState, self.__position_jaw_current_cb)
        ]

        # create node
        if not rospy.get_node_uri():
            rospy.init_node(
                'dvrkArm_node', anonymous=True, log_level=rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() +
                           ' -> ROS already initialized')
        self.__interval_ms = 30  # Sept 6: Minho suggests 20ms --> 30ms?
        self.__rate = rospy.Rate(1000.0 / self.__interval_ms)
        while True:
            if self.__act_pos==[] or self.__act_rot==[] or self.__act_jaw==[] or self.__act_joint==[]:
                pass
            else:
                break

    """
    Callback function
    """

    def __goal_reached_cb(self, data):
        self.__goal_reached = data.data
        self.__goal_reached_event.set()

    def __position_cartesian_current_cb(self, data):
        self.__act_pose_frame = posemath.fromMsg(data.pose)
        self.__act_pos = [
            data.pose.position.x, data.pose.position.y, data.pose.position.z
        ]
        self.__act_rot = [
            data.pose.orientation.x, data.pose.orientation.y,
            data.pose.orientation.z, data.pose.orientation.w
        ]
        self.__get_position_event.set()

    def __position_joint_current_cb(self, data):
        self.__act_joint = list(data.position)
        self.__get_joint_event.set()

    def __position_jaw_current_cb(self, data):
        self.__act_jaw = list(data.position)
        self.__get_jaw_event.set()

    """
    Get function
    """

    def get_current_pose_frame(self):
        return self.__act_pose_frame

    def get_current_orientation(self):
        return self.__get_rot_transform(self.__act_rot)

    def get_current_orientation_quaternion(self):
        return self.__act_rot

    def get_current_position(self, wait_callback=False):
        if wait_callback:
            self.__get_position_event.clear()
            if self.__get_position_event.wait(20):  # 20 seconds at most
                return self.__act_pos
            else:
                return []
        else:
            return self.__act_pos

    def get_current_joint(self, wait_callback=False):
        if wait_callback:
            self.__get_joint_event.clear()
            if self.__get_joint_event.wait(20):  # 20 seconds at most
                joint = self.__act_joint
                return joint
            else:
                return []
        else:
            joint = self.__act_joint
            return joint

    def get_current_jaw(self, wait_callback=False):
        if wait_callback:
            self.__get_jaw_event.clear()
            if self.__get_jaw_event.wait(20):  # 20 seconds at most
                jaw = self.__act_jaw
                return jaw
            else:
                return []
        else:
            jaw = self.__act_jaw
            return jaw

    """
    Set function
    """

    def set_pose(self, pos=[], rot=[], wait_callback=True):
        msg = Pose()
        if not rot == []:
            rot_transformed = self.__set_rot_transform(rot)

        if pos == []:
            msg.position.x = self.__act_pos[0]
            msg.position.y = self.__act_pos[1]
            msg.position.z = self.__act_pos[2]
        else:
            msg.position.x = pos[0]
            msg.position.y = pos[1]
            msg.position.z = pos[2]

        if rot == []:
            msg.orientation.x = self.__act_rot[0]
            msg.orientation.y = self.__act_rot[1]
            msg.orientation.z = self.__act_rot[2]
            msg.orientation.w = self.__act_rot[3]
        else:
            msg.orientation.x = rot_transformed[0]
            msg.orientation.y = rot_transformed[1]
            msg.orientation.z = rot_transformed[2]
            msg.orientation.w = rot_transformed[3]

        if wait_callback:
            self.__goal_reached_event.clear()
            self.__set_position_goal_cartesian_pub.publish(msg)
            return self.__goal_reached_event.wait(10)  # 10 seconds at most:
        else:
            self.__set_position_goal_cartesian_pub.publish(msg)
            return True

    def set_jaw(self, jaw, wait_callback=True):
        msg = JointState()
        msg.position = jaw
        if wait_callback:
            self.__goal_reached_event.clear()
            self.__set_position_goal_jaw_pub.publish(msg)
            return self.__goal_reached_event.wait(10)  # 10 seconds at most
        else:
            self.__set_position_goal_jaw_pub.publish(msg)
            return True

    # specify intermediate points between q0 & qf using linear interpolation (blocked until goal reached)
    def set_pose_linear(self, pos, rot):
        q0 = self.get_current_position(wait_callback=True)
        qf = pos
        assert len(qf) > 0, qf
        assert len(q0) > 0, q0

        pos_rtol = 1.e-5
        pos_atol = 1.e-3
        if np.allclose(q0, qf, pos_rtol, pos_atol):
            return False
        else:
            tf = np.linalg.norm(np.array(qf) - np.array(q0))**0.8 * 10
            v_limit = (np.array(qf) - np.array(q0)) / tf
            v = v_limit * 1.5
            # print '\n'
            # print 'q0=', q0
            # print 'qf=', qf
            # print 'norm=', np.linalg.norm(np.array(qf) - np.array(q0))
            # print 'tf=', tf
            # print 'v=',v
            t = 0.0
            while True:
                q = self.__LSPB(q0, qf, t, tf, v)
                self.set_pose(q, rot, wait_callback=False)
                t += 0.001 * self.__interval_ms
                self.__rate.sleep()
                if t > tf:
                    break

    def inverse_kin_arm(self, pos):
        L1 = 0.4318  # Lrcc
        L2 = 0.4162  # Ltool
        L3 = 0.0091  # Lpitch2yaw
        L4 = 0.0102  # Lyaw2tip

        # Forward Kinematics
        # x = np.cos(q2)*np.sin(q1)*(L2-L1+L3+L4+q3)
        # y = -np.sin(q2)*(L2-L1+L3+L4+q3)
        # z = -np.cos(q1)*np.cos(q2)*(L2-L1+L3+L4+q3)
        x = pos[0]
        y = pos[1]
        z = pos[2]

        # Inverse Kinematics
        j1 = np.arctan2(x, -z)
        j2 = np.arctan2(-y, np.sqrt(x**2 + z**2))
        j3 = np.sqrt(x**2 + y**2 + z**2) + L1 - L2 - L3 - L4
        return j1, j2, j3

    def set_arm_position(self, pos):
        j1, j2, j3 = self.inverse_kin_arm(pos)
        self.set_joint([j1, j2, j3, 0.0, 0.0, 0.0], True)

    def set_joint(self, joint, wait_callback=True):
        msg = JointState()
        msg.position = joint
        if wait_callback:
            self.__goal_reached_event.clear()
            self.__set_position_goal_joint_pub.publish(msg)
            return self.__goal_reached_event.wait(10)  # 20 seconds at most
        else:
            self.__set_position_goal_joint_pub.publish(msg)
            return True

    """
    Conversion function
    """

    # Matching coordinate of the robot base and the end effector
    def __set_rot_transform(self, q):
        qx, qy, qz, qw = q
        R1 = PyKDL.Rotation.Quaternion(qx, qy, qz, qw)
        R2 = PyKDL.Rotation.EulerZYX(-np.pi / 2, 0,
                                     0)  # rotate -90 (deg) around z-axis
        R3 = PyKDL.Rotation.EulerZYX(0, np.pi,
                                     0)  # rotate 180 (deg) around y-axis
        R = R1 * R2 * R3
        return R.GetQuaternion()

    # Matching coordinate of the robot base and the end effector
    def __get_rot_transform(self, q):
        qx, qy, qz, qw = q
        R1 = PyKDL.Rotation.Quaternion(qx, qy, qz, qw)
        R2 = PyKDL.Rotation.EulerZYX(0, np.pi,
                                     0)  # rotate 180 (deg) around y-axis
        R3 = PyKDL.Rotation.EulerZYX(-np.pi / 2, 0,
                                     0)  # rotate -90 (deg) around z-axis
        R = R1 * R2.Inverse() * R3.Inverse()
        return R.GetQuaternion()

    """
    Trajectory
    """

    def __LSPB(self, q0, qf, t, tf, v):

        if np.allclose(q0, qf): return q0
        elif np.all(v) == 0: return q0
        elif tf == 0: return q0
        elif tf < 0: return []
        elif t < 0: return []
        else:
            v_limit = (np.array(qf) - np.array(q0)) / tf
            if np.allclose(U.normalize(v), U.normalize(v_limit)):
                if np.linalg.norm(v) < np.linalg.norm(
                        v_limit) or np.linalg.norm(
                            2 * v_limit) < np.linalg.norm(v):
                    return []
                else:
                    tb = np.linalg.norm(
                        np.array(q0) - np.array(qf) +
                        np.array(v) * tf) / np.linalg.norm(v)
                    a = np.array(v) / tb
                    if 0 <= t and t < tb:
                        q = np.array(q0) + np.array(a) / 2 * t * t
                    elif tb < t and t <= tf - tb:
                        q = (np.array(qf) + np.array(q0) -
                             np.array(v) * tf) / 2 + np.array(v) * t
                    elif tf - tb < t and t <= tf:
                        q = np.array(qf) - np.array(a) * tf * tf / 2 + np.array(
                            a) * tf * t - np.array(a) / 2 * t * t
                    else:
                        return []
                    return q
            else:
                return []


if __name__ == "__main__":
    p1 = dvrkArm('/PSM1')
    p2 = dvrkArm('/PSM2')

    # Set position
    # pos = [0.1, 0.0, -0.15]
    # rot = [0, 0, 0]
    # q = U.euler_to_quaternion(rot, unit='deg')
    # p1.set_pose(pos=pos, rot=q)
    # jaw = [0*np.pi/180.]
    # p1.set_jaw(jaw)
    # p2.set_jaw(jaw)

    # Set joint
    # joint = [0.0, 0.0, 0.15, -1.0, -0.5, -0.5]
    # p1.set_joint(joint, True)

    # Set position with straightened wrist
    # p1.set_position([-0.05, 0.0, -0.15])
    # p2.set_position([0.05, 0.0, -0.15])

    pos1 = [0.0, 0.0, -0.13]
    rot1 = [0.0, 0.0, 0.0]
    q1 = U.euler_to_quaternion(rot1, unit='deg')
    jaw1 = [20 * np.pi / 180.]
    pos2 = [0.0, 0.0, -0.13]
    rot2 = [0.0, 0.0, 0.0]
    q2 = U.euler_to_quaternion(rot2, unit='deg')
    jaw2 = [0.0]
    joint1 = [0.9, 0.0, 0.15, 1.0, 0.0, 0.0]
    joint2 = [0.0, 0.0, 0.13, 0.0, 0.0, 0.0]

    rot1[0] = 0
    rot1[2] = 50
    rot1[1] = -30
    q1 = U.euler_to_quaternion(rot1, unit='deg')
    p1.set_pose(rot=q1)
    p1.set_jaw(jaw1)

    # t = 0.0
    # while True:
    #     t += 0.01
    #     joint1[4] = 0.8*np.sin(2*3.14*t)
    #     joint1[5] = 0.8*np.sin(2*3.14*t)
    #     p1.set_joint(joint1)
    #     # rot1[1] = 60*np.sin(2*3.14*t*3)
    #     # q1 = U.euler_to_quaternion(rot1, unit='deg')
    #     # p1.set_pose(rot=q1)
    #     # p1.set_jaw(jaw1)
