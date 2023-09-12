import os.path
import random
from math import atan, tan, pi, dist

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.io import read_image

SAMPLES_N = 20
RATIO = 1000 / 1024 * 0.9
DIA_THRESHOLD = 50

def init(img_nparr):
    # img_path = os.path.join(img_path)
    # # img_path_output = os.path.join(img_output_dir, img_name)
    # img_tensor = read_image(img_path)
    #
    # img_pil = TF.to_pil_image(img_tensor)

    # img_array = np.asarray(img_pil)

    num_labels, labels = cv2.connectedComponents(img_nparr)

    return num_labels, labels


# 丢弃小的区域
def drop_small_area(labels, num_labels, threshold=2000):
    num_labels_arr = []
    for i in range(1, num_labels):
        label_part = labels == i
        if sum(sum(label_part)) >= threshold:
            num_labels_arr.append(i)
    return num_labels_arr


def find_dia(labels, num_labels_arr, preserve_ratio=1 / 3):
    # ans = []
    img = np.copy(labels)
    inner_points_list = []
    outer_points_list = []
    global_max_dia = 0
    global_sum_dia = 0
    # print(num_labels)

    global_dia_large_num = 0


    for label in num_labels_arr:
        label_img = np.argwhere(labels == label)

        x_down = np.max(label_img[:, 0])
        x_up = max(0, np.min(label_img[:, 0]) - 10)
        y_left = np.min(label_img[:, 1])
        y_right = np.max(label_img[:, 1])

        x_down = int(x_up + (x_down - x_up) * preserve_ratio)
        x_area = [x_up, x_down]
        y_area = [y_left, y_right]
        # # 选择区域
        # for i in range(x_up, x_down):
        #     img[i, y_left] = 50
        #     img[i, y_right] = 50
        # for i in range(y_left, y_right):
        #     img[x_up, i] = 50
        #     img[x_down, i] = 50

        # 采样SAMPLES_N个点
        x_linespace = np.linspace(x_up, x_down, SAMPLES_N, dtype=int)
        inner_points_y = []
        for x in x_linespace:

            tem = []  # 横截的坐标点
            for y in range(y_left, y_right):
                # 异或，只能有一边为血管，另外一边为背景
                if labels[x, y] == label and (labels[x, y - 1] == 0) ^ (labels[x, y + 1] == 0):
                    tem.append(y)

            inner_points_y.append(tem)
        target_pos = []

        for i in range(SAMPLES_N):
            if len(inner_points_y[i]) != 4:
                continue

            # 这里存的是[x, y]
            target_pos.append([x_linespace[i], np.mean(inner_points_y[i][1:3], dtype=int)])

        if len(target_pos) < 2:
            continue

        # pos1 是上面的那个点
        pos1 = target_pos[0]
        pos2 = target_pos[1]

        # 生长方向
        std_radian = atan((pos2[1] - pos1[1]) / (pos2[0] - pos1[0]))

        # 内侧最顶点
        inner_pos = grow(std_radian, pos1, labels, x_area, y_area, label)
        if not inner_pos:
            continue

        # outer_points = find_outer_points_v1(inner_pos, img, x_area, y_area)
        outer_points = find_outer_points_v2(inner_pos, labels, label, ratio=0.15)
        if len(outer_points) == 0:
            continue

        # 画出所有待选区域
        # draw_points(outer_points, img)
        max_dis_outer_point, dia_len = calc_dis(inner_pos, outer_points)

        inner_points_list.append(inner_pos)
        outer_points_list.append(max_dis_outer_point)
        global_max_dia = max(global_max_dia, dia_len)
        global_sum_dia += dia_len

        if dia_len >= DIA_THRESHOLD:
            global_dia_large_num += 1

    # if len(inner_points_list) == 0:
    #     global_mean_dia = 0
    # else:
    #     global_mean_dia = np.round(global_sum_dia / len(inner_points_list), 4)

    return img, inner_points_list, outer_points_list, global_max_dia, global_sum_dia, global_dia_large_num


def find_outer_points_v2(inner_points, labels, label, ratio=0.0):
    # 找到基于inner_point 的外侧outer_points的左右y_left和y_right
    y_slices = np.where(labels[inner_points[0], :] == label)
    y_left = np.min(y_slices)
    y_right = np.max(y_slices)

    y_left, y_right = y_left + (y_right - y_left) * ratio, y_right - (y_right - y_left) * ratio

    target_points = np.argwhere(labels == label)
    mask = (target_points[:, 1] > y_left) & (target_points[:, 1] < y_right)
    target_points = target_points[mask]
    unique_y, y_index = np.unique(target_points[:, 1], return_index=True)

    # 使用minimum.reduceat函数计算每个y轴值的最小x轴，并使用unique_y在结果中索引
    result_x = np.minimum.reduceat(target_points[:, 0], y_index)[np.searchsorted(target_points[y_index, 1], unique_y)]

    # 将unique_y和result_x合并成一个新的2维数组，其中每行包含一个y轴值和该y值对应的最小x轴
    result = np.column_stack((result_x, unique_y))

    # print(result)

    return result


def draw_point(pos1, image, color=50):
    image[pos1[0], pos1[1]] = color


def draw_points(points, image, color=50):
    indexes = tuple(points.T)
    image[indexes] = color


def draw_2points_line(pos1, pos2, image, color=50):
    k = (pos2[1] - pos1[1]) / (pos2[0] - pos1[0])

    # 保证pos1在上面，即x更小
    if pos1[0] > pos2[0]:
        pos1, pos2 = pos2, pos1
    pos_x = pos2[0]
    pos_y = pos2[1]
    while pos_x > pos1[0]:
        pos_x -= 1
        pos_y -= k
        image[pos_x, round(pos_y)] = color


# 按照当前位置pos，以k=tan(g_angel)进行生长，直到出现值为target_label的点出现
def grow(g_angel, start_pos, labels, x_limit, y_limit, target_label=0):
    pos_x = start_pos[0]
    pos_y = start_pos[1]
    k = tan(g_angel)
    while x_limit[0] <= pos_x <= x_limit[1] and y_limit[0] <= pos_y <= y_limit[1] and \
            labels[pos_x, round(pos_y)] != target_label:
        pos_x -= 1
        pos_y -= k
    if x_limit[0] <= pos_x <= x_limit[1] and y_limit[0] <= pos_y <= y_limit[1]:
        return [pos_x, round(pos_y)]
    return []


def generate_radian(sample_num=10):
    return np.linspace(-pi / 4,
                       pi / 4,
                       sample_num)


def calc_dis(inner_point, outer_points):
    # 将inner_point扩展成与outer_points相同形状的数组
    point = np.array(inner_point)
    point = np.tile(point, (len(outer_points), 1))

    distances = np.round(np.sqrt(np.sum((outer_points - point) ** 2, axis=1)) * RATIO, 4)
    print("distances", distances)
    print("DIA_THRESHOLD", DIA_THRESHOLD)

    # 找到距离最大的点的索引
    max_index = np.argmax(distances)

    # 获取距离最大的点的坐标和距离
    max_point = outer_points[max_index]
    max_distance = distances[max_index]

    print("inner_point:", inner_point, "距离最大的点是", max_point, "，距离为", max_distance)
    return max_point, max_distance


def save_output(name,
                img_output_dir=r"D:\Projects\medical_prediction_py_service\disease_segmentation_version5\dia_output"):
    img_path_output = os.path.join(img_output_dir, name)
    plt.savefig(img_path_output)
    plt.close()
    return img_path_output


def draw_save(img, inner_points, outer_points, name, plt_show_flag=False):
    plt.imshow(img, cmap='viridis')
    colors = ['#3682be', '#45a776', '#f05326', '#eed777', '#334f65', '#b3974e', '#38cb7d', '#ddae33', '#844bb3',
              '#93c555',
              '#5f6694', '#df3881']
    #
    for i in range(len(inner_points)):
        color = random.choice(colors)
        inner_p = transform_coordinate(inner_points[i])
        outer_p = transform_coordinate(outer_points[i])

        plt.plot([inner_p[0], outer_p[0]], [inner_p[1], outer_p[1]], '-', color=color)
        dia = round(dist(inner_p, outer_p) * RATIO, 1)
        plt.annotate(dia, xy=((inner_p[0] + outer_p[0]) / 2, (inner_p[1] + outer_p[1]) / 2), xytext=(5, 5),
                     textcoords='offset points', fontsize=10, color=color)
    file_path = save_output(name)

    if plt_show_flag:
        plt.show()

    return file_path


def transform_coordinate(pos):
    pos[0], pos[1] = pos[1], pos[0]
    return pos


def main(np_arr):
    num_labels, labels = init(np_arr)
    num_labels_arr = drop_small_area(labels, num_labels, threshold=1200)
    img, inner_points_list, outer_points_list, max_dia, mean_dia, large_dia_num = find_dia(labels, num_labels_arr, 0.5)
    all_vessel_num = len(num_labels_arr)
    # calc_vessel_num = len(outer_points_list)
    calc_vessel_num = large_dia_num
    return img, inner_points_list, outer_points_list, all_vessel_num, calc_vessel_num, max_dia, mean_dia


if __name__ == '__main__':
    img_name = "43CAPorg3_mask.png"
    img_tensor = read_image(img_name)
    img_pil = TF.to_pil_image(img_tensor)
    img_array = np.asarray(img_pil)

    img, inner_points_list, outer_points_list, vessel_all, vessel_calc, max_dia, mean_dia = main(img_array)
    print(vessel_all)
    print(vessel_calc)

    file_path = draw_save(img, inner_points_list, outer_points_list, img_name, plt_show_flag=True)
    print(max_dia)
    print(mean_dia)
