from dvrkMotionBridgeP import dvrkMotionBridgeP
from dvrkKinematics import dvrkKinematics
from ZividCapture import ZividCapture
import utils as U
import numpy as np


# Define objects
dvrk = dvrkMotionBridgeP()
zivid = ZividCapture(initialize=True)

# Set pose of the end effector
pos = [0.07, 0.07, -0.13]   # position in (m)
rot = [0.0, 0.0, 0.0]   # Euler angles
quat = U.euler_to_quaternion(rot, unit='deg')   # convert Euler angles to quaternion
jaw = [0*np.pi/180.]    # jaw angle in (rad)
dvrk.set_pose(pos1=pos, rot1=quat, jaw1=jaw)

# Set joint of the end effector
joint = [0.8, -0.3, 0.12, 0.0, 0.0,  0.0]   # in (rad) or (m)
jaw = [0*np.pi/180.]    # jaw angle in (rad)
dvrk.set_joint(joint1=joint, jaw1=jaw)

# Get pose of the end effector
pos, rot, jaw = dvrk.get_pose()
print (pos, rot, jaw)

# Get joint of the end effector
joint = dvrk.get_joint()
print (joint)

# Capture and display RGB-D images
# img_color = zivid.capture_2Dimage(color='RGB')  # faster capture of RGB image (20~90 fps)
img_color, img_depth, img_point = zivid.capture_3Dimage(color='RGB')    # 7~10 fps
zivid.display_rgb(img_color)
zivid.display_depthmap(img_point)
zivid.display_pointcloud(img_point, img_color)