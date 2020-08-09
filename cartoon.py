import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

data_set = np.array([[.9, .9], [.85, 2.1], [1.2, 1.], [2.1, .95], [3., 1.1],
                     [3.9, .7], [4., 1.4], [4.2, 1.8], [2., 2.3], [3., 2.3],
                     [1.5, 1.8], [2., 1.5], [2.2, 2.], [2.6, 1.7], [2.7,
                                                                    1.85]])
categories = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
color1 = (0.69411766529083252, 0.3490196168422699, 0.15686275064945221, 1.0)
color2 = (0.65098041296005249, 0.80784314870834351, 0.89019608497619629, 1.0)
colormap = np.array([color1, color2])
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.scatter(
#     x=[data_set[:, 0]],
#     y=[data_set[:, 1]],
#     c=colormap[categories],
#     marker='o',
#     alpha=0.9
# )

margin = .1
min_f0, max_f0 = -70, 20
min_f1, max_f1 = 5, 10
width = max_f0 - min_f0
height = max_f1 - min_f1

ax.add_patch(
    patches.Rectangle(
        xy=(min_f0, min_f1),  # point of origin.
        width=width,
        height=height,
        linewidth=1,
        color='red',
        fill=True))

margin = .1
min_f0, max_f0 = -70, 20
min_f1, max_f1 = -10, -5
width = max_f0 - min_f0
height = max_f1 - min_f1

ax.add_patch(
    patches.Rectangle(
        xy=(min_f0, min_f1),  # point of origin.
        width=width,
        height=height,
        linewidth=1,
        color='red',
        fill=True))

circle = plt.Circle((-50, 0), radius=1)
ax.add_patch(circle)
circle = plt.Circle((0, 0), radius=1)
ax.add_patch(circle)
label = ax.annotate("start", xy=(-50, 3), fontsize=10, ha="center")
label = ax.annotate("goal", xy=(0, 3), fontsize=10, ha="center")

plt.xlim(-60, 10)
plt.ylim(-10, 10)

ax.set_aspect('equal')
ax.autoscale_view()

plt.savefig("pointbot0_cartoon.png")

plt.show()
