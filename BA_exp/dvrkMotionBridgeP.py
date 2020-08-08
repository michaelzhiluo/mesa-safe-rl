import socket
import struct
import numpy as np
import threading
import BA_exp.utils as U


class dvrkMotionBridgeP():
    def __init__(self):
        # Data members
        self.des_pos1 = []
        self.des_rot1 = []
        self.des_jaw1 = []
        self.des_pos2 = []
        self.des_rot2 = []
        self.des_jaw2 = []
        self.des_joint1 = []
        self.des_joint2 = []
        self.act_pos1 = []
        self.act_rot1 = []
        self.act_jaw1 = []
        self.act_pos2 = []
        self.act_rot2 = []
        self.act_jaw2 = []
        self.act_joint1 = []
        self.act_joint2 = []
        self.input_flag = [True, True, True, True, True, True, True, True]

        # UDP setting
        self.UDP_IP = "127.0.0.1"
        self.UDP_PORT_SERV = 1215
        self.UDP_PORT_SERV2 = 1216  # auxiliary channel for sending actual value
        self.UDP_PORT_CLNT = 1217
        self.UDP_PORT_CLNT2 = 1218
        self.buffer_size = 1024
        self.sock = socket.socket(
            socket.AF_INET,  # Internet
            socket.SOCK_DGRAM)  # UDP
        self.sock2 = socket.socket(
            socket.AF_INET,  # Internet
            socket.SOCK_DGRAM)  # UDP
        self.sock.setblocking(1)  # Blocking mode
        self.sock.bind((self.UDP_IP, self.UDP_PORT_CLNT))
        self.sock2.bind((self.UDP_IP, self.UDP_PORT_CLNT2))

        # Thread
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        while True:
            data_recv, addr = self.sock2.recvfrom(
                self.buffer_size)  # buffer size is 1024 bytes
            data_unpack = list(struct.unpack('=28f', data_recv))
            self.act_pos1 = data_unpack[0:3]
            self.act_rot1 = data_unpack[3:7]
            self.act_jaw1 = data_unpack[7:8]
            self.act_pos2 = data_unpack[8:11]
            self.act_rot2 = data_unpack[11:15]
            self.act_jaw2 = data_unpack[15:16]
            self.act_joint1 = data_unpack[16:22]
            self.act_joint2 = data_unpack[22:28]

    def send_motion_data(self, func_numb):
        if self.des_pos1 == []:
            self.des_pos1 = [0.0, 0.0, -0.13]
            self.input_flag[0] = False
        if self.des_rot1 == []:
            self.des_rot1 = [0.0, 0.0, 0.0, 1.0]
            self.input_flag[1] = False
        if self.des_jaw1 == []:
            self.des_jaw1 = [0.0]
            self.input_flag[2] = False
        if self.des_pos2 == []:
            self.des_pos2 = [0.0, 0.0, -0.13]
            self.input_flag[3] = False
        if self.des_rot2 == []:
            self.des_rot2 = [0.0, 0.0, 0.0, 1.0]
            self.input_flag[4] = False
        if self.des_jaw2 == []:
            self.des_jaw2 = [0.0]
            self.input_flag[5] = False
        if self.des_joint1 == []:
            self.des_joint1 = [0.0, 0.0, 0.13, 0.0, 0.0, 0.0]
            self.input_flag[6] = False
        if self.des_joint2 == []:
            self.des_joint2 = [0.0, 0.0, 0.13, 0.0, 0.0, 0.0]
            self.input_flag[7] = False

        concat = list(self.des_pos1)+list(self.des_rot1)+list(self.des_jaw1)\
                 +list(self.des_pos2)+list(self.des_rot2)+list(self.des_jaw2)\
                 +list(self.des_joint1)+list(self.des_joint2)
        data_send = struct.pack('=%sf8?i' % len(concat), *concat,
                                *self.input_flag, int(func_numb))
        self.sock.sendto(data_send, (self.UDP_IP, self.UDP_PORT_SERV))

        # data receiving
        data, _ = self.sock.recvfrom(self.buffer_size)
        goal_reached = list(struct.unpack('?', data))
        self.set_default()
        return goal_reached[0]

    def set_pose(self, pos1=[], rot1=[], jaw1=[], pos2=[], rot2=[], jaw2=[]):
        self.des_pos1 = pos1
        self.des_rot1 = rot1
        self.des_jaw1 = jaw1
        self.des_pos2 = pos2
        self.des_rot2 = rot2
        self.des_jaw2 = jaw2
        return self.send_motion_data(0)

    def set_joint(self, joint1=[], jaw1=[], joint2=[], jaw2=[]):
        self.des_joint1 = joint1
        self.des_jaw1 = jaw1
        self.des_joint2 = joint2
        self.des_jaw2 = jaw2
        return self.send_motion_data(1)

    def set_arm_position(self, pos1=[], pos2=[]):
        self.des_pos1 = pos1
        self.des_pos2 = pos2
        return self.send_motion_data(2)

    def set_default(self):
        self.des_pos1 = []
        self.des_rot1 = []
        self.des_jaw1 = []
        self.des_pos2 = []
        self.des_rot2 = []
        self.des_jaw2 = []
        self.des_joint1 = []
        self.des_joint2 = []
        self.input_flag = [True, True, True, True, True, True, True, True]

    def get_pose(self):
        return self.act_pos1, self.act_rot1, self.act_jaw1

    def get_joint(self):
        return self.act_joint1


if __name__ == "__main__":
    perception = dvrkMotionBridgeP()
    pos1 = [0.07, 0.07, -0.13]
    pos2 = [0.03, 0.03, -0.13]
    rot1 = [0.0, 0.0, 0.0]
    q1 = U.euler_to_quaternion(rot1, unit='deg')
    jaw1 = [0 * np.pi / 180.]
    # perception.set_pose(jaw1=jaw1)
    import time
    while True:
        perception.set_pose(pos1=pos1, rot1=q1, jaw1=jaw1)
        time.sleep(1)
        perception.set_pose(pos1=pos2, rot1=q1, jaw1=jaw1)
        time.sleep(1)

    # pos2 = [0.0, 0.0, -0.13]
    # rot2 = [0.0, 0.0, 0.0]
    # q2 = U.euler_to_quaternion(rot2, unit='deg')
    # jaw2 = [5*np.pi/180.]
    # # perception.set_pose(jaw1=jaw1)
    # joint1 = [0.8, .7, 0.12, 0.0, 0.0,  0.0]
    # joint2 = [0.8, .7, 0.12, 0.0, -0.5, 0.0]

    # perception.set_joint(joint1=joint1)
    # t = 0.0
    # import time
    # while True:
    #     print(perception.act_joint1)
    #     t += 0.005
    #     pos1[0] = 0.1*np.sin(2*3.14*t)
    #     pos2[0] = 0.1*np.sin(2*3.14*t)
    #     jaw1[0] = 0.6*np.sin(2*3.14*2*t) + 0.6
    #     jaw2[0] = 0.6*np.sin(2*3.14*2*t) + 0.6
    #     perception.set_arm_position(pos1=pos1, pos2=pos2)
    #     perception.set_pose(pos1=pos1, rot1=q1, jaw1=jaw1, pos2=pos2, rot2=q2, jaw2=jaw2)
    #     # perception.set_position(pos1=pos1, pos2=pos2)
    #     # joint1[0] = 1.13 + 0.1*np.sin(2*3.14*t)
    #     # joint1[1] = -0.10 + 0.1*np.sin(2*3.14*t)
    #     # joint1[2] = 0.16 + 0.003*np.sin(2*3.14*t)
    #     # joint1[3] = 1.3 + 0.9*np.sin(2*3.14*t)
    #     # joint1[4] = -0.5434 + 0.5*np.sin(3*3.14*t)
    #     # joint1[5] = 0.4066 + 0.3*np.sin(3*3.14*t)
    #     # perception.set_joint(joint1)
    #     # time.sleep(0.005)
