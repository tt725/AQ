import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd

exp_idx = 0
units = dict()
def get_dataset(algo_path, base_path):
    global exp_idx
    global units

    dataset = {}
    for key in algo_path.keys():
        complete_path = os.path.join(base_path, algo_path[key])
        temp_data = []
        for root, _, files in os.walk(complete_path):
            if 'progress.txt' in files:
                data = pd.read_table(os.path.join(root, 'progress.txt'))

                condition = key
                exp_idx += 1
                if condition not in units:
                    units[condition] = 0
                unit = units[condition]
                units[condition] += 1

                performance = 'AverageTestEpRet' if 'AverageTestEpRet' in data else 'AverageEpRet'
                data.insert(len(data.columns), 'Unit', unit)
                data.insert(len(data.columns), 'Condition', condition)
                if performance in data:
                    data.insert(len(data.columns), 'Performance', data[performance])
                temp_data.append(data)
        dataset[key] = temp_data
    exp_idx = 0
    units = dict()
    return dataset

def figure(hopper_sota, ant_sota, walk_sota,Hopper_AQ_a):
    fig = plt.figure(figsize=(13, 3.5))
    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect1 = [0.05, 0.26, 0.18, 0.68]
    rect2 = [0.3, 0.26, 0.18, 0.68]
    rect3 = [0.55, 0.26, 0.18, 0.68]
    rect4 = [0.8, 0.26, 0.18, 0.68]
    ax1 = plt.axes(rect1)
    ax2 = plt.axes(rect2)
    ax3 = plt.axes(rect3)
    ax4 = plt.axes(rect4)


    color_sota = {
        'DDPG': 'blue',
        'TD3': 'black',
        'AvgDQN': 'green',
        'REDQ': 'c',
        'AdaEQ': 'm',
        'AdaADDPG': 'red',
    }

    print("Hopper sota")
    for key in hopper_sota.keys():
        data_list = []
        data_list.append(hopper_sota[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(30)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax1.plot(xx, smoothed_x, label=f"{key}", color=color_sota[key])
        print(smoothed_x[-1])
    ax1.set_ylabel("Average return", fontsize=15)
    ax1.set_xlabel("Steps", fontsize=15)
    ax1.set_xlim(0, 3e5)
    ax1.set_ylim(0, 3500)
    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(1e5)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.xaxis.set_major_locator(xx)
    ax1.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax1.xaxis.get_offset_text().set_fontsize(15)
    ax1.set_title(label=r'(a) SOTA in Hopper', fontsize=15, y=-0.4)
    ax1.grid()
    ax1.legend(fontsize=12, loc="upper left", handlelength=1)


    print("Ant SOTA")
    for key in ant_sota.keys():
        data_list = []
        data_list.append(ant_sota[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(50)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax2.plot(xx, smoothed_x, label=f"{key}", color=color_sota[key])
        print(smoothed_x[-1])
    ax2.set_ylabel("Average return", fontsize=15)
    ax2.set_xlabel("Steps", fontsize=15)
    ax2.set_xlim(0, 5e5)
    ax2.set_ylim(0, 3500)
    ax2.tick_params(labelsize=15)
    xx = MultipleLocator(2e5)
    ax2.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax2.yaxis.set_major_locator(yy)
    ax2.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax2.yaxis.get_offset_text().set_fontsize(15)
    ax2.xaxis.set_major_locator(xx)
    ax2.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax2.xaxis.get_offset_text().set_fontsize(15)
    ax2.set_title(label=r'(b) SOTA in Ant', fontsize=15, y=-0.4)
    ax2.grid()
    ax2.legend(fontsize=12, loc="upper left", handlelength=1)

    print("Walk sota")
    for key in walk_sota.keys():
        data_list = []
        data_list.append(walk_sota[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(100)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax3.plot(xx, smoothed_x, label=f"{key}", color=color_sota[key])
        print(smoothed_x[-1])
    ax3.set_ylabel("Average return", fontsize=15)
    ax3.set_xlabel("Steps", fontsize=15)
    ax3.set_xlim(0, 1e6)
    ax3.set_ylim(0, 5000)
    ax3.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax3.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2000)
    ax3.yaxis.set_major_locator(yy)
    ax3.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax3.yaxis.get_offset_text().set_fontsize(15)
    ax3.xaxis.set_major_locator(xx)
    ax3.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax3.xaxis.get_offset_text().set_fontsize(15)
    ax3.set_title(label=r'(c) SOTA in Walker2d', fontsize=15, y=-0.4)
    ax3.grid()
    ax3.legend(fontsize=12, loc="upper left", handlelength=1)

    color_a = {
        'DDPG': 'blue',
        'TD3': 'black',
        'AdaADDPG(1)': 'red',
        'AdaADDPG(2)': 'green',
        'AdaADDPG(5)': 'c',
        'AdaADDPG(10)': 'm',
        'AdaADDPG(100)': 'y'
    }

    print("Hopper_AQ_a")
    for key in Hopper_AQ_a.keys():
        data_list = []
        data_list.append(Hopper_AQ_a[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(30)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax4.plot(xx, smoothed_x, label=f"{key}", color=color_a[key])
        print(smoothed_x[-1])
    ax4.set_ylabel("Average return", fontsize=15)
    ax4.set_xlabel("Steps", fontsize=15)
    ax4.set_xlim(0, 3e5)
    ax4.set_ylim(0, 3500)
    ax4.tick_params(labelsize=15)
    xx = MultipleLocator(1e5)
    ax4.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax4.yaxis.set_major_locator(yy)
    ax4.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax4.yaxis.get_offset_text().set_fontsize(15)
    ax4.xaxis.set_major_locator(xx)
    ax4.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax4.xaxis.get_offset_text().set_fontsize(15)
    ax4.set_title(label=r'(d) AdaADDPG($\tau$) in Hopper', fontsize=15, y=-0.4)
    ax4.grid()
    ax4.legend(fontsize=12, loc="upper left", handlelength=1)


    plt.savefig("./DeepC.pdf", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()


if __name__ == "__main__":

    base_path_Hopper = '../data/Hopper'
    algo_path_Hopper = {
        'DDPG': 'Hopper-v2_ddpg_N1',
        'TD3': 'Hopper-v2_TD3_N2',
        'AvgDQN': 'Hopper-v2_average_N2',
        'REDQ': 'Hopper-v2_redq_N10_M2',
        'AdaEQ': 'Hopper-v2_adaeq_N10_M2_C0.3',
        'AdaADDPG': 'Hopper-v2_SoftAQ_N2_T1',
    }
    hopper_sota = {}
    hopper_sota.update(get_dataset(algo_path_Hopper, base_path_Hopper))

    base_path_Ant = '../data/Ant'
    algo_path_Ant = {
        'DDPG': 'Ant-v2_ddpg_N1',
        'TD3': 'Ant-v2_TD3_N2',
        'AvgDQN': 'Ant-v2_average_N2',
        'REDQ': 'Ant-v2_redq_N10_M2',
        'AdaEQ': 'Ant-v2_adaeq_N10_M2_C0.3',
        'AdaADDPG': 'Ant-v2_SoftAQ_N2_T1',
    }
    ant_sota = {}
    ant_sota.update(get_dataset(algo_path_Ant, base_path_Ant))

    base_path_Walker2d = '../data/Walker2d'
    algo_path_Walker2d = {
        'DDPG': 'Walker2d-v2_ddpg_N1',
        'TD3': 'Walker2d-v2_TD3_N2',
        'AvgDQN': 'Walker2d-v2_average_N2',
        'REDQ': 'Walker2d-v2_redq_N10_M2',
        'AdaEQ': 'Walker2d-v2_adaeq_N10_M2_C0.3',
        'AdaADDPG': 'Walker2d-v2_SoftAQ_N2_T1',
    }
    walk_sota = {}
    walk_sota.update(get_dataset(algo_path_Walker2d, base_path_Walker2d))

    base_path_Hopper_AQ_Over = '../data/Hopper'
    algo_path_Hopper_AQ_Over = {
        'DDPG': 'Hopper-v2_ddpg_N1',
        'TD3': 'Hopper-v2_TD3_N2',
        'AdaADDPG(1)': 'Hopper-v2_SoftAQ_N2_T1',
        'AdaADDPG(2)': 'Hopper-v2_SoftAQ_N2_T2',
        'AdaADDPG(5)': 'Hopper-v2_SoftAQ_N2_T5',
        'AdaADDPG(10)': 'Hopper-v2_SoftAQ_N2_T10',
        'AdaADDPG(100)': 'Hopper-v2_SoftAQ_N2_T100',
    }
    Hopper_AQ_a = {}
    Hopper_AQ_a.update(get_dataset(algo_path_Hopper_AQ_Over, base_path_Hopper_AQ_Over))



    figure(hopper_sota, ant_sota, walk_sota,Hopper_AQ_a)
