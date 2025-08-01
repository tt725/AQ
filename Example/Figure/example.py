import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import ast
import numpy as np
from scipy.interpolate import make_interp_spline


def list_subtract(list1, list2):
    result = list1.copy()
    for item in list2:
        if item in result:
            result.remove(item)
    return result


def figure(multiarm_m):
    fig = plt.figure(figsize=(6.5, 3.5))

    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect1 = [0.1, 0.26, 0.36, 0.66]
    rect2 = [0.6, 0.26, 0.36, 0.66]
    ax1 = plt.axes(rect1)
    ax2 = plt.axes(rect2)

    label_m = [r'Q-learning', ]
    x_value = [i for i in range(len(multiarm_m[0]))]
    ax1.plot(x_value, multiarm_m[0], linewidth=3.0, label=label_m[0], color='blue')
    ax1.set_ylabel(r'$\gamma \max_{a} Q_{t}(S,a)$', fontsize=15)
    ax1.set_xlabel(r'$t (x 100)$', fontsize=15)
    ax1.set_xlim(0, len(multiarm_m[0]))
    ax1.set_ylim(0, 5)
    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.grid()
    ax1.legend(fontsize=11, loc="upper left", handlelength=3)
    ax1.set_title(label=r'(a) Overestimation bias', fontsize=15, y=-0.4)

    gamma=0.95
    learningRate=0.5
    data = multiarm_m[1]
    print(data)
    thresholds = [
        [],
        [data[1] - data[0] * learningRate * gamma],
        [data[2] - data[0] * learningRate * gamma * (1 - learningRate) - data[1] * learningRate * gamma,
         data[2] - data[0] * learningRate * gamma * (1 - learningRate)],
        [data[3] - data[0] * learningRate * gamma * (1 - learningRate) ** 2 - data[1] * learningRate * gamma * (
                    1 - learningRate) - data[
             2] * learningRate * gamma,
         data[3] - data[0] * learningRate * gamma * (1 - learningRate) ** 2 - data[1] * learningRate * gamma * (
                     1 - learningRate),
         data[3] - data[0] * learningRate * gamma * (1 - learningRate) ** 2],
        [data[4] - data[0] * learningRate * gamma * (1 - learningRate) ** 3 - data[1] * learningRate * gamma * (
                1 - learningRate) ** 2 - data[2] * learningRate * gamma * (1 - learningRate) - data[
             3] * learningRate * gamma,
         data[4] - data[0] * learningRate * gamma * (1 - learningRate) ** 3 - data[1] * learningRate * gamma * (
                 1 - learningRate) ** 2 - data[2] * learningRate * gamma * (1 - learningRate),
         data[4] - data[0] * learningRate * gamma * (1 - learningRate) ** 3 - data[1] * learningRate * gamma * (
                 1 - learningRate) ** 2,
         data[4] - data[0] * learningRate * gamma * (1 - learningRate) ** 3]
    ]

    print(thresholds)

    x = np.arange(len(data))
    width = 0.3

    for i, line_positions in enumerate(thresholds):
        for y in line_positions:
            ax2.plot([x[i] - width / 2, x[i] + width / 2], [y, y], color='black', linestyle='--', linewidth=1.5)

    keypoint_base = [data[0],
                     data[1] - data[0] * learningRate * gamma,
                     data[2] - data[0] * learningRate * gamma * (1 - learningRate) - data[1] * learningRate * gamma,
                     data[3] - data[0] * learningRate * gamma * (1 - learningRate) ** 2 - data[
                         1] * learningRate * gamma * (1 - learningRate) -
                     data[2] * learningRate * gamma,
                     data[4] - data[0] * learningRate * gamma * (1 - learningRate) ** 3 - data[
                         1] * learningRate * gamma * (1 - learningRate) ** 2 -
                     data[2] * learningRate * gamma * (1 - learningRate) - data[3] * learningRate * gamma
                     ]
    print(keypoint_base)
    ax2.bar(x, keypoint_base, width=width, facecolor='none', edgecolor='black', hatch='//',
            label=r'cur-bias')

    ax2.bar(x, data, width=width, facecolor='none', edgecolor='black', hatch='',
            label=r'prop-bias')

    x_smooth = np.linspace(x.min(), x.max(), 200)
    spl = make_interp_spline(x, keypoint_base, k=3)
    y_smooth = spl(x_smooth)
    ax2.plot(x_smooth, y_smooth, color='red', linestyle='--', linewidth=1.5)


    ax2.set_ylabel(r'$\gamma \max_{a} Q_{t}(S,a)$', fontsize=15)
    ax2.set_xlabel(r'$t (x 1)$', fontsize=15)
    ax2.set_ylim(0, 1.6)
    ax2.tick_params(labelsize=15)
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    yy = MultipleLocator(0.5)
    ax2.yaxis.set_major_locator(yy)
    ax2.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax2.yaxis.get_offset_text().set_fontsize(15)
    ax2.legend(fontsize=11, loc="upper left", handlelength=3)
    ax2.set_title(label=r'(b) Cur & Prop-bias', fontsize=15, y=-0.4)

    plt.savefig("./Example.pdf", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()
    

def get_Q1_value(dir):
    log = open(dir, 'r').readlines()
    max_Q_S = log[-1][:]
    max_Q_S = ast.literal_eval(max_Q_S)
    return max_Q_S

def get_Q2_value(dir):
    log = open(dir, 'r').readlines()
    max_Q_S_5 = log[-3][:]
    max_Q_S_5 = ast.literal_eval(max_Q_S_5)
    return max_Q_S_5


if __name__ == "__main__":
    Q1_learning_dir = r'../Result/log.Q 2025.06.11.16.19.20'
    y1_value = get_Q1_value(
        dir=Q1_learning_dir)

    y2_value = get_Q2_value(dir=Q1_learning_dir)

    multiarm_m = [y1_value, y2_value]

    figure(multiarm_m)