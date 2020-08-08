import socket
import struct
import threading, time
from .dvrkDualArm import dvrkDualArm


class dvrkMotionBridgeM():
    def __init__(self):
        self.dvrk = dvrkDualArm()

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
        self.sock.bind((self.UDP_IP, self.UDP_PORT_SERV))
        self.sock2.bind((self.UDP_IP, self.UDP_PORT_SERV2))

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        self.main()

    def run(self):
        while True:
            # actual positions
            act_pos1 = self.dvrk.arm1._dvrkArm__act_pos
            act_rot1 = self.dvrk.arm1._dvrkArm__act_rot
            act_jaw1 = self.dvrk.arm1._dvrkArm__act_jaw
            act_pos2 = self.dvrk.arm2._dvrkArm__act_pos
            act_rot2 = self.dvrk.arm2._dvrkArm__act_rot
            act_jaw2 = self.dvrk.arm2._dvrkArm__act_jaw
            act_joint1 = self.dvrk.arm1._dvrkArm__act_joint
            act_joint2 = self.dvrk.arm2._dvrkArm__act_joint
            concat = list(act_pos1) + list(act_rot1) + list(act_jaw1) + list(
                act_pos2) + list(act_rot2) + list(act_jaw2) + list(
                    act_joint1) + list(act_joint2)
            data_send = struct.pack('=%sf' % len(concat), *concat)
            self.sock2.sendto(data_send, (self.UDP_IP, self.UDP_PORT_CLNT2))
            time.sleep(0.001)  # 1 ms

    def main(self):
        while True:
            # data receiving
            data_recv, _ = self.sock.recvfrom(self.buffer_size)
            data_unpack = list(struct.unpack('=28f8?i', data_recv))
            des_pos1 = data_unpack[0:3]
            des_rot1 = data_unpack[3:7]
            des_jaw1 = data_unpack[7:8]
            des_pos2 = data_unpack[8:11]
            des_rot2 = data_unpack[11:15]
            des_jaw2 = data_unpack[15:16]
            des_joint1 = data_unpack[16:22]
            des_joint2 = data_unpack[22:28]
            if data_unpack[28] == False: des_pos1 = []
            if data_unpack[29] == False: des_rot1 = []
            if data_unpack[30] == False: des_jaw1 = []
            if data_unpack[31] == False: des_pos2 = []
            if data_unpack[32] == False: des_rot2 = []
            if data_unpack[33] == False: des_jaw2 = []
            if data_unpack[34] == False: des_joint1 = []
            if data_unpack[35] == False: des_joint2 = []
            method = data_unpack[36]
            if method == 0:
                self.dvrk.set_pose(
                    pos1=des_pos1, rot1=des_rot1, pos2=des_pos2, rot2=des_rot2)
                self.dvrk.set_jaw(jaw1=des_jaw1, jaw2=des_jaw2)
            elif method == 1:
                self.dvrk.set_joint(joint1=des_joint1, joint2=des_joint2)
                self.dvrk.set_jaw(jaw1=des_jaw1, jaw2=des_jaw2)
            elif method == 2:
                self.dvrk.set_arm_position(pos1=des_pos1, pos2=des_pos2)

            # data sending
            data_send = struct.pack('?', True)
            self.sock.sendto(data_send, (self.UDP_IP, self.UDP_PORT_CLNT))
            time.sleep(0.001)  # 1 ms loop


if __name__ == "__main__":
    motion = dvrkMotionBridgeM()
