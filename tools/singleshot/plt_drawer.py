"""
Create: 2026.05.09
Author: SG.SUH
Python: 3.8.13
PyTorch: 1.12.0
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg

class plt_drawer:
    def __init__(self,
                 img_size=(1600, 900)):
        self.img_size = img_size

    def draw_plot(self,
                  img_data,
                  bev_objlist,
                  cam_objlist):
        plt.cla()

        fig, axes = plt.subplots(2, 1, figsize=(5, 5))
        axes[0].plot(0, 0, "x", color="red")
        self.draw_bbox(axes[0], bev_objlist, [0, 40], [-20, 20])

        axes[0].set_axisbelow(True)
        axes[0].xaxis.grid(color="gray", linestyle="dashed")
        axes[0].yaxis.grid(color="gray", linestyle="dashed")
        axes[0].set_xlim(0, 40)
        axes[0].set_ylim(-20, 20)

        axes[1].imshow(img_data)

        self.draw_bbox(axes[1], cam_objlist, [0, self.img_size[0]], [0, self.img_size[1]])

        axes[1].set_xlim(0, self.img_size[0])
        axes[1].set_ylim(self.img_size[1], 0)
        axes[1].axis("off")

    def draw_bbox(self,
                  ax,
                  detected_obj_list,
                  limit_x,
                  limit_y):
        if detected_obj_list is None:
            return
        
        for objinfo in detected_obj_list:
            linewidth = 0.5
            color = objinfo.color
            corners = objinfo.corners
            alpha_obj = 0.7
            alpha_txt = 0.5

            if corners is None or math.isnan(corners[0][0]):
                continue

            center_bottom_forward = np.mean(corners.T[2:4], axis=0)
            center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)

            ax.plot([center_bottom[0], center_bottom_forward[0]], [center_bottom[1], center_bottom_forward[1]], color=color, linewidth=linewidth, alpha=alpha_obj)

            min_point = np.min(corners.T, axis=0)

            if limit_x[0] < min_point[0] < limit_x[1] - 5 and limit_y[0] < min_point[1] < limit_y[1]:
                ax.text(min_point[0], min_point[1], f"[{objinfo.id_}]", bbox={"facecolor": color, "edgecolor": color, "alpha": alpha_txt, "pad": 0}, fontdict={"size": 5}, weight="bold")

            for idx in range(4):
                ax.plot([corners.T[idx][0], corners.T[idx + 4][0]], [corners.T[idx][1], corners.T[idx + 4][1]], color=color, linewidth=linewidth, alpha=alpha_obj)

            def draw_rect(selected_corners,
                          color):
                prev = selected_corners[-1]

                for corner in selected_corners:
                    ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth, alpha=alpha_obj)

                    prev = corner

            draw_rect(corners.T[:4], color)
            draw_rect(corners.T[4:], color)

    def save_fig(self,
                 output_filename):
        plt.savefig(output_filename, bbox_inches="tight", pad_inches=0, dpi=200)