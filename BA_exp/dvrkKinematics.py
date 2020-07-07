from math import pi
import numpy as np
import utils as U
import BA_exp.dvrkVariables as dvrkVar

class dvrkKinematics():
    def __init__(self):
        pass

    @classmethod
    def pose_to_transform(cls, pos, rot):
        """

        :param pos: position (m)
        :param rot: quaternion (qx, qy, qz, qw)
        :return:
        """
        T = np.zeros((4, 4))
        R = U.quaternion_to_R(rot[0], rot[1], rot[2], rot[3])
        T[:3,:3] = R
        T[:3,-1] = np.transpose(pos)
        T[-1,-1] = 1
        return T

    def pose_to_joint(self, pos, rot, method='analytic'):
        if pos==[] or rot==[]:
            joint = []
        else:
            T = np.matrix(self.pose_to_transform(pos, rot))    # current transformation
            joint = self.ik(T, method)
        return joint

    @classmethod
    def transform(cls, a, alpha, d, theta, unit='rad'):    # modified DH convention
        if unit=='deg':
            alpha = np.deg2rad(alpha)
            theta = np.deg2rad(theta)
        T = np.array([[np.cos(theta), -np.sin(theta), 0, a],
                      [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                      [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha),  np.cos(alpha)*d],
                      [0, 0, 0, 1]])
        return T

    @classmethod
    def fk(cls, joints, L1=0, L2=0, L3=0, L4=0):
        q1, q2, q3, q4, q5, q6 = joints
        T01 = dvrkKinematics.transform(0, np.pi/2, 0, q1+np.pi/2)
        T12 = dvrkKinematics.transform(0, -np.pi/2, 0, q2-np.pi/2)
        T23 = dvrkKinematics.transform(0, np.pi/2, q3-L1+L2, 0)
        T34 = dvrkKinematics.transform(0, 0, 0, q4)
        T45 = dvrkKinematics.transform(0, -np.pi/2, 0, q5-np.pi/2)
        T56 = dvrkKinematics.transform(L3, -np.pi/2, 0, q6-np.pi/2)
        T67 = dvrkKinematics.transform(0, -np.pi/2, L4, 0)
        T78 = dvrkKinematics.transform(0, np.pi, 0, np.pi)
        T08 = T01@T12@T23@T34@T45@T56@T67@T78
        return T08

    def ik(self, T, method='analytic'):
        if method=='analytic':
            T = np.linalg.inv(T)
            x84 = T[0, 3]
            y84 = T[1, 3]
            z84 = T[2, 3]
            q6 = np.arctan2(x84, z84 - dvrkVar.L4)
            temp = -dvrkVar.L3 + np.sqrt(x84 ** 2 + (z84 - dvrkVar.L4) ** 2)
            q4 = dvrkVar.L1 - dvrkVar.L2 + np.sqrt(y84 ** 2 + temp ** 2)
            q5 = np.arctan2(-y84, temp)
            R84 = np.array([[np.sin(q5) * np.sin(q6), -np.cos(q6), np.cos(q5) * np.sin(q6)],
                            [np.cos(q5), 0, -np.sin(q5)],
                            [np.cos(q6) * np.sin(q5), np.sin(q6), np.cos(q5) * np.cos(q6)]])
            R80 = T[:3, :3]
            R40 = R84.T @ R80
            n32 = R40[2, 1]
            n31 = R40[2, 0]
            n33 = R40[2, 2]
            n22 = R40[1, 1]
            n12 = R40[0, 1]
            q2 = np.arcsin(n32)
            q1 = np.arctan2(-n31, n33)
            q3 = np.arctan2(n22, n12)
            joint = [q1, q2, q4, q3, q5, q6]  # q3 and q4 are swapped
        elif method=='numerical':
            q0 = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # initial guess
            ik_sol = self.ikine(T, q0)
            joint = [ik_sol[0, 0], ik_sol[0, 1], ik_sol[0, 2], ik_sol[0, 3], ik_sol[0, 4], ik_sol[0, 5]]
        assert ~np.isnan(joint).any()
        return joint

    @classmethod
    def ik_position_wrist(cls, pos, L1=0.4318, L2=0.4162):
        # L1: Rcc (m)
        # L2: length of tool (m)
        x = pos[0]      # (m)
        y = pos[1]
        z = pos[2]

        # Forward Kinematics
        # x = np.cos(q2)*np.sin(q1)*(L2-L1+q3)
        # y = -np.sin(q2)*(L2-L1+q3)
        # z = -np.cos(q1)*np.cos(q2)*(L2-L1+q3)

        # Inverse Kinematics
        q1 = np.arctan2(x, -z)     # (rad)
        q2 = np.arctan2(-y, np.sqrt(x ** 2 + z ** 2))  # (rad)
        q3 = np.sqrt(x ** 2 + y ** 2 + z ** 2) + L1 - L2  # (m)
        return q1, q2, q3

    @classmethod
    def ik_orientation(cls, q1,q2,Rb):
        R03 = np.array([[-np.sin(q1)*np.sin(q2), -np.cos(q1), np.cos(q2)*np.sin(q1)],
                        [-np.cos(q2), 0, -np.sin(q2)],
                        [np.cos(q1)*np.sin(q2), -np.sin(q1), -np.cos(q1)*np.cos(q2)]])
        R38 = R03.T.dot(Rb)
        r12 = R38[0,1]
        r22 = R38[1,1]
        r31 = R38[2,0]
        r32 = R38[2,1]
        r33 = R38[2,2]
        q4 = np.arctan2(-r22, -r12)     # (rad)
        q6 = np.arctan2(-r31, -r33)
        q5 = np.arctan2(r32, np.sqrt(r31**2+r33**2))
        return q4,q5,q6