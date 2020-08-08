"""Shared methods, to be loaded in other code.
"""
import numpy as np

ESC_KEYS = [27, 1048603]
MILLION = float(10**6)


def normalize(v):
    norm = np.linalg.norm(v, ord=2)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def LPF(raw_data, fc, dt):
    filtered = np.zeros_like(raw_data)
    for i in range(len(raw_data)):
        if i == 0:
            filtered[0] = raw_data[0]
        else:
            filtered[i] = 2 * np.pi * fc * dt * raw_data[i] + (
                1 - 2 * np.pi * fc * dt) * filtered[i - 1]
    return filtered


def euler_to_quaternion(rot, unit='rad'):
    if unit == 'deg':
        rot = np.deg2rad(rot)

    # for the various angular functions
    z, y, x = rot
    cy = np.cos(z * 0.5)
    sy = np.sin(z * 0.5)
    cp = np.cos(y * 0.5)
    sp = np.sin(y * 0.5)
    cr = np.cos(x * 0.5)
    sr = np.sin(x * 0.5)

    # quaternion
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr
    return [qx, qy, qz, qw]


def quaternion_to_eulerAngles(q, unit='rad'):
    qx, qy, qz, qw = q

    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if (abs(sinp) >= 1):
        pitch = np.sign(sinp) * (np.pi / 2)
        # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    if unit == 'deg':
        [roll, pitch, yaw] = np.rad2deg([roll, pitch, yaw])
    return [roll, pitch, yaw]


def quaternion_to_R(qx, qy, qz, qw):
    s = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    r11 = 1 - 2 * s * (qy * qy + qz * qz)
    r12 = 2 * s * (qx * qy - qz * qw)
    r13 = 2 * s * (qx * qz + qy * qw)
    r21 = 2 * s * (qx * qy + qz * qw)
    r22 = 1 - 2 * s * (qx * qx + qz * qz)
    r23 = 2 * s * (qy * qz - qx * qw)
    r31 = 2 * s * (qx * qz - qy * qw)
    r32 = 2 * s * (qy * qz + qx * qw)
    r33 = 1 - 2 * s * (qx * qx + qy * qy)
    R = [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
    return R


def R_to_euler(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def R_to_quaternion(R):
    raise NotImplementedError
    qw = np.sqrt((1. + R[0][0] + R[1][1] + R[2][2]) / 2.)
    qx = (R[2][1] - R[1][2]) / (4. * qw)
    qy = (R[0][2] - R[2][0]) / (4. * qw)
    qz = (R[1][0] - R[0][1]) / (4. * qw)
    return qx, qy, qz, qw


# Get a rigid transformation matrix from pts1 to pts2
def get_rigid_transform(pts1, pts2):
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    mean1 = pts1.mean(axis=0)
    mean2 = pts2.mean(axis=0)
    pts1 = np.array([p - mean1 for p in pts1])
    pts2 = np.array([p - mean2 for p in pts2])
    # if option=='clouds':
    H = pts1.T.dot(pts2)  # covariance matrix
    U, S, V = np.linalg.svd(H)
    V = V.T
    R = V.dot(U.T)
    t = -R.dot(mean1.T) + mean2.T
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, -1] = t
    T[-1, -1] = 1
    return T


def minor(arr, i, j):
    # ith row, jth column removed
    arr = np.array(arr)
    return arr[np.array(list(range(i)) +
                        list(range(i + 1, arr.shape[0])))[:, np.newaxis],
               np.array(list(range(j)) + list(range(j + 1, arr.shape[1])))]


def create_waveform(data_range, amp1, amp2, amp3, amp4, freq1, freq2, freq3,
                    freq4, phase, step):
    t = np.arange(0, 1, 1.0 / step)
    waveform1 = amp1 * np.sin(2 * np.pi * freq1 * (t - phase))
    waveform2 = amp2 * np.sin(2 * np.pi * freq2 * (t - phase))
    waveform3 = amp3 * np.sin(2 * np.pi * freq3 * (t - phase))
    waveform4 = amp4 * np.sin(2 * np.pi * freq4 * (t - phase))
    waveform = waveform1 + waveform2 + waveform3 + waveform4
    x = waveform / max(waveform)
    y = (data_range[1] - data_range[0]) / 2.0 * x + (
        data_range[1] + data_range[0]) / 2.0
    return t, y


if __name__ == '__main__':
    # calculate_transformation()
    # filename = '/home/hwangmh/pycharmprojects/FLSpegtransfer/vision/coordinate_pairs.npy'
    # data = np.load(filename)
    # print(data)

    pts1 = [[0, 1, 0], [1, 0, 0], [0, -1, 0]]
    pts2 = [[-0.7071, 0.7071, 0], [0.7071, 0.7071, 0], [0.7071, -0.7071, 0]]
    T = get_rigid_transform(pts1, pts2)
    print(T)

    # f = 6  # (Hz)
    # A = 1  # amplitude
    # t, waveform = create_waveform(interp=[0.1, 0.5], amp1=A, amp2=A * 3, amp3=A * 4, freq1=f, freq2=f * 1.8,
    #                               freq3=f * 1.4, phase=0.0, step=200)
    # t, waveform = create_waveform(interp=[0.1, 0.5], amp1=A, amp2=A * 1.2, amp3=A * 4.2, freq1=0.8 * f, freq2=f * 1.9,
    #                               freq3=f * 1.2, phase=0.5, step=200)
    # t, waveform = create_waveform(interp=[0.1, 0.5], amp1=A, amp2=A * 1.5, amp3=A * 3.5, freq1=f, freq2=f * 1.8,
    #                               freq3=f * 1.3, phase=0.3, step=200)
