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

def figure(Hopper_m, Hopper_n, Hopper_c, Hopper_a, ant_m, ant_n, ant_c, ant_a, walk_m, walk_n, walk_c, walk_a):
    fig = plt.figure(figsize=(13, 10.5))
    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect1 = [0.05, 0.76, 0.18, 0.22]
    rect2 = [0.3, 0.76, 0.18, 0.22]
    rect3 = [0.55, 0.76, 0.18, 0.22]
    rect4 = [0.8, 0.76, 0.18, 0.22]
    rect5 = [0.05, 0.42, 0.18, 0.22]
    rect6 = [0.3, 0.42, 0.18, 0.22]
    rect7 = [0.55, 0.42, 0.18, 0.22]
    rect8 = [0.8, 0.42, 0.18, 0.22]
    rect9 = [0.05, 0.08, 0.18, 0.22]
    rect10 = [0.3, 0.08, 0.18, 0.22]
    rect11 = [0.55, 0.08, 0.18, 0.22]
    rect12 = [0.8, 0.08, 0.18, 0.22]
    ax1 = plt.axes(rect5)
    ax2 = plt.axes(rect6)
    ax4 = plt.axes(rect7)
    ax3 = plt.axes(rect8)
    ax5 = plt.axes(rect9)
    ax6 = plt.axes(rect10)
    ax8 = plt.axes(rect11)
    ax7 = plt.axes(rect12)
    ax9 = plt.axes(rect1)
    ax10 = plt.axes(rect2)
    ax12 = plt.axes(rect4)
    ax11 = plt.axes(rect3)


    color_m = {
        'DDPG': 'blue',
        'ADDPG(1,4)':  'green',
        'ADDPG(2,4)':  'c',
        'ADDPG(4,4)':  'm',
        'ADDPG(8,4)':  'y',
        'ADDPG(16,4)': '#1f77b4'
    }

    print("Ant_m")
    for key in ant_m.keys():
        data_list = []
        data_list.append(ant_m[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(50)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax1.plot(xx, smoothed_x, label=f"{key}", color=color_m[key])
        print(smoothed_x[-1])

    ax1.set_ylabel("Average return", fontsize=15)
    ax1.set_xlabel("Steps", fontsize=15)
    ax1.set_xlim(0, 5e5)
    ax1.set_ylim(0, 3500)
    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(2e5)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.xaxis.set_major_locator(xx)
    ax1.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax1.xaxis.get_offset_text().set_fontsize(15)
    ax1.set_title(label=r'(e) ADDPG$(M,4)$ in Ant', fontsize=15, y=-0.4)
    ax1.grid()
    ax1.legend(fontsize=12, loc="upper left", handlelength=1)


    color_n = {
        'TD3': 'black',
        'ADDPG(4,1)': 'green',
        'ADDPG(4,2)': 'c',
        'ADDPG(4,4)': 'm',
        'ADDPG(4,8)': 'y',
        'ADDPG(4,16)': 'red',
    }

    print("Ant_n")
    for key in ant_n.keys():
        data_list = []
        data_list.append(ant_n[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(50)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax2.plot(xx, smoothed_x, label=f"{key}", color=color_n[key])
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
    ax2.set_title(label=r'(f) ADDPG$(4,N)$ in Ant', fontsize=15, y=-0.4)
    ax2.grid()
    ax2.legend(fontsize=12, loc="upper left", handlelength=1)

    color_c = {
        'DDPG': 'blue',
        'TD3': 'black',
        'ADDPG(4,4)': 'green',
        'AdaADDPG(1)': 'red'
    }

    print("Ant_c")
    for key in ant_c.keys():
        data_list = []
        data_list.append(ant_c[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(50)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax3.plot(xx, smoothed_x, label=f"{key}", color=color_c[key])
        print(smoothed_x[-1])
    ax3.set_ylabel("Average return", fontsize=15)
    ax3.set_xlabel("Steps", fontsize=15)
    ax3.set_xlim(0, 5e5)
    ax3.set_ylim(0, 3500)
    ax3.tick_params(labelsize=15)
    xx = MultipleLocator(2e5)
    ax3.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax3.yaxis.set_major_locator(yy)
    ax3.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax3.yaxis.get_offset_text().set_fontsize(15)
    ax3.xaxis.set_major_locator(xx)
    ax3.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax3.xaxis.get_offset_text().set_fontsize(15)
    ax3.set_title(label=r'(h) AdaADDPG in Ant', fontsize=15, y=-0.4)
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

    print("Ant_a")
    for key in ant_a.keys():
        data_list = []
        data_list.append(ant_a[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(50)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax4.plot(xx, smoothed_x, label=f"{key}", color=color_a[key])
        print(smoothed_x[-1])
    ax4.set_ylabel("Average return", fontsize=15)
    ax4.set_xlabel("Steps", fontsize=15)
    ax4.set_xlim(0, 5e5)
    ax4.set_ylim(0, 3500)
    ax4.tick_params(labelsize=15)
    xx = MultipleLocator(2e5)
    ax4.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax4.yaxis.set_major_locator(yy)
    ax4.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax4.yaxis.get_offset_text().set_fontsize(15)
    ax4.xaxis.set_major_locator(xx)
    ax4.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax4.xaxis.get_offset_text().set_fontsize(15)
    ax4.set_title(label=r'(g) AdaADDPG$(\tau)$ in Ant', fontsize=15, y=-0.4)
    ax4.grid()
    ax4.legend(fontsize=12, loc="upper left", handlelength=1)


    print("Walk_m")
    for key in walk_m.keys():
        data_list = []
        data_list.append(walk_m[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(100)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax5.plot(xx, smoothed_x, label=f"{key}", color=color_m[key])
        print(smoothed_x[-1])
    ax5.set_ylabel("Average return", fontsize=15)
    ax5.set_xlabel("Steps", fontsize=15)
    ax5.set_xlim(0, 1e6)
    ax5.set_ylim(0, 5000)
    ax5.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax5.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2000)
    ax5.yaxis.set_major_locator(yy)
    ax5.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax5.yaxis.get_offset_text().set_fontsize(15)
    ax5.xaxis.set_major_locator(xx)
    ax5.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax5.xaxis.get_offset_text().set_fontsize(15)
    ax5.set_title(label=r'(i) ADDPG$(M,4)$ in Walker2d', fontsize=15, y=-0.4)
    ax5.grid()
    ax5.legend(fontsize=12, loc="upper left", handlelength=1)


    print("Walk_n")
    for key in walk_n.keys():
        data_list = []
        data_list.append(walk_n[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(100)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax6.plot(xx, smoothed_x, label=f"{key}", color=color_n[key])
        print(smoothed_x[-1])
    ax6.set_ylabel("Average return", fontsize=15)
    ax6.set_xlabel("Steps", fontsize=15)
    ax6.set_xlim(0, 1e6)
    ax6.set_ylim(0, 5000)
    ax6.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax6.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2000)
    ax6.yaxis.set_major_locator(yy)
    ax6.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax6.yaxis.get_offset_text().set_fontsize(15)
    ax6.xaxis.set_major_locator(xx)
    ax6.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax6.xaxis.get_offset_text().set_fontsize(15)
    ax6.set_title(label=r'(j) ADDPG$(4,N)$ in Walker2d', fontsize=15, y=-0.4)
    ax6.grid()
    ax6.legend(fontsize=12, loc="upper left", handlelength=1)


    print("walk_c")
    for key in walk_c.keys():
        data_list = []
        data_list.append(walk_c[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(100)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax7.plot(xx, smoothed_x, label=f"{key}", color=color_c[key])
        print(smoothed_x[-1])
    ax7.set_ylabel("Average return", fontsize=15)
    ax7.set_xlabel("Steps", fontsize=15)
    ax7.set_xlim(0, 1e6)
    ax7.set_ylim(0, 5000)
    ax7.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax7.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2000)
    ax7.yaxis.set_major_locator(yy)
    ax7.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax7.yaxis.get_offset_text().set_fontsize(15)
    ax7.xaxis.set_major_locator(xx)
    ax7.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax7.xaxis.get_offset_text().set_fontsize(15)
    ax7.set_title(label=r'(l) AdaADDPG in Walker2d', fontsize=15, y=-0.4)
    ax7.grid()
    ax7.legend(fontsize=12, loc="upper left", handlelength=1)

    print("Walk_a")
    for key in walk_a.keys():
        data_list = []
        data_list.append(walk_a[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(100)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax8.plot(xx, smoothed_x, label=f"{key}", color=color_a[key])
        print(smoothed_x[-1])
    ax8.set_ylabel("Average return", fontsize=15)
    ax8.set_xlabel("Steps", fontsize=15)
    ax8.set_xlim(0, 1e6)
    ax8.set_ylim(0, 5000)
    ax8.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax8.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2000)
    ax8.yaxis.set_major_locator(yy)
    ax8.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax8.yaxis.get_offset_text().set_fontsize(15)
    ax8.xaxis.set_major_locator(xx)
    ax8.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax8.xaxis.get_offset_text().set_fontsize(15)
    ax8.set_title(label=r'(k) AdaADDPG$(\tau)$ in Walker2d', fontsize=15, y=-0.4)
    ax8.grid()
    ax8.legend(fontsize=12, loc="upper left", handlelength=1)


    print("Hopper_m")
    for key in Hopper_m.keys():
        data_list = []
        data_list.append(Hopper_m[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(30)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax9.plot(xx, smoothed_x, label=f"{key}", color=color_m[key])
        print(smoothed_x[-1])

    ax9.set_ylabel("Average return", fontsize=15)
    ax9.set_xlabel("Steps", fontsize=15)
    ax9.set_xlim(0, 3e5)
    ax9.set_ylim(0, 3500)
    ax9.tick_params(labelsize=15)
    xx = MultipleLocator(1e5)
    ax9.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax9.yaxis.set_major_locator(yy)
    ax9.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax9.yaxis.get_offset_text().set_fontsize(15)
    ax9.xaxis.set_major_locator(xx)
    ax9.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax9.xaxis.get_offset_text().set_fontsize(15)
    ax9.set_title(label=r'(a) ADDPG$(M,4)$ in Hopper', fontsize=15, y=-0.4)
    ax9.grid()
    ax9.legend(fontsize=12, loc="upper left", handlelength=1)


    print("Hopper_n")
    for key in Hopper_n.keys():
        data_list = []
        data_list.append(Hopper_n[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(30)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax10.plot(xx, smoothed_x, label=f"{key}", color=color_n[key])
        print(smoothed_x[-1])
    ax10.set_ylabel("Average return", fontsize=15)
    ax10.set_xlabel("Steps", fontsize=15)
    ax10.set_xlim(0, 3e5)
    ax10.set_ylim(0, 3500)
    ax10.tick_params(labelsize=15)
    xx = MultipleLocator(1e5)
    ax10.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax10.yaxis.set_major_locator(yy)
    ax10.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax10.yaxis.get_offset_text().set_fontsize(15)
    ax10.xaxis.set_major_locator(xx)
    ax10.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax10.xaxis.get_offset_text().set_fontsize(15)
    ax10.set_title(label=r'(b) ADDPG$(4,N)$ in Hopper', fontsize=15, y=-0.4)
    ax10.grid()
    ax10.legend(fontsize=12, loc="upper left", handlelength=1)

    print("Hopper_c")
    for key in Hopper_c.keys():
        data_list = []
        data_list.append(Hopper_c[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(30)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax12.plot(xx, smoothed_x, label=f"{key}", color=color_c[key])
        print(smoothed_x[-1])
    ax12.set_ylabel("Average return", fontsize=15)
    ax12.set_xlabel("Steps", fontsize=15)
    ax12.set_xlim(0, 3e5)
    ax12.set_ylim(0, 3500)
    ax12.tick_params(labelsize=15)
    xx = MultipleLocator(1e5)
    ax12.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax12.yaxis.set_major_locator(yy)
    ax12.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax12.yaxis.get_offset_text().set_fontsize(15)
    ax12.xaxis.set_major_locator(xx)
    ax12.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax12.xaxis.get_offset_text().set_fontsize(15)
    ax12.set_title(label=r'(d) AdaADDPG in Hopper', fontsize=15, y=-0.4)
    ax12.grid()
    ax12.legend(fontsize=12, loc="upper left", handlelength=1)


    print("Hopper_a")
    for key in Hopper_a.keys():
        data_list = []
        data_list.append(Hopper_a[key])
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
        grouped_data = data_combined.groupby("TotalEnvInteracts").mean()
        y = np.ones(30)
        x = np.asarray(grouped_data["Performance"].copy())
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        xx = grouped_data.index
        ax11.plot(xx, smoothed_x, label=f"{key}", color=color_a[key])
        print(smoothed_x[-1])
    ax11.set_ylabel("Average return", fontsize=15)
    ax11.set_xlabel("Steps", fontsize=15)
    ax11.set_xlim(0, 3e5)
    ax11.set_ylim(0, 3500)
    ax11.tick_params(labelsize=15)
    xx = MultipleLocator(1e5)
    ax11.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax11.yaxis.set_major_locator(yy)
    ax11.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax11.yaxis.get_offset_text().set_fontsize(15)
    ax11.xaxis.set_major_locator(xx)
    ax11.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax11.xaxis.get_offset_text().set_fontsize(15)
    ax11.set_title(label=r'(c) AdaADDPG$(\tau)$ in Hopper', fontsize=15, y=-0.4)
    ax11.grid()
    ax11.legend(fontsize=12, loc="upper left", handlelength=1)


    plt.savefig("./DeepCSupple.pdf", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()


if __name__ == "__main__":
    base_path_Hopper_AQ_Over = '../data/Hopper'
    algo_path_Hopper_AQ_Over = {
        'DDPG': 'Hopper-v2_ddpg_N1',
        'ADDPG(1,4)': 'Hopper-v2_AQ_N2_O8_U4',
        'ADDPG(2,4)': 'Hopper-v2_AQ_N2_O1_U4',
        'ADDPG(4,4)': 'Hopper-v2_AQ_N2_O4_U16',
        'ADDPG(8,4)': 'Hopper-v2_AQ_N2_O2_U4',
        'ADDPG(16,4)': 'Hopper-v2_AQ_N2_O16_U4',
    }
    Hopper_m = {}
    Hopper_m.update(get_dataset(algo_path_Hopper_AQ_Over, base_path_Hopper_AQ_Over))

    base_path_Hopper_AQ_Over = '../data/Hopper'
    algo_path_Hopper_AQ_Over = {
        'TD3': 'Hopper-v2_TD3_N2',
        'ADDPG(4,1)': 'Hopper-v2_AQ_N2_O4_U1',
        'ADDPG(4,2)': 'Hopper-v2_AQ_N2_O4_U2',
        'ADDPG(4,4)': 'Hopper-v2_AQ_N2_O4_U16',
        'ADDPG(4,8)': 'Hopper-v2_AQ_N2_O4_U8',
        'ADDPG(4,16)': 'Hopper-v2_AQ_N2_O4_U4',
    }
    Hopper_n = {}
    Hopper_n.update(get_dataset(algo_path_Hopper_AQ_Over, base_path_Hopper_AQ_Over))

    base_path_Hopper_AQ_Over = '../data/Hopper'
    algo_path_Hopper_AQ_Over = {
        'DDPG': 'Hopper-v2_ddpg_N1',
        'TD3': 'Hopper-v2_TD3_N2',
        'ADDPG(4,4)': 'Hopper-v2_AQ_N2_O4_U16',
        'AdaADDPG(1)': 'Hopper-v2_SoftAQ_N2_T1',
    }
    Hopper_c = {}
    Hopper_c.update(get_dataset(algo_path_Hopper_AQ_Over, base_path_Hopper_AQ_Over))

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
    Hopper_a = {}
    Hopper_a.update(get_dataset(algo_path_Hopper_AQ_Over, base_path_Hopper_AQ_Over))

    base_path_Hopper_AQ_Over = '../data/Ant'
    algo_path_Hopper_AQ_Over = {
        'DDPG': 'Ant-v2_ddpg_N1',
        'ADDPG(1,4)': 'Ant-v2_AQ_N2_O8_U4',
        'ADDPG(2,4)': 'Ant-v2_AQ_N2_O2_U4',
        'ADDPG(4,4)': 'Ant-v2_AQ_N2_O4_U16',
        'ADDPG(8,4)': 'Ant-v2_AQ_N2_O4_U4',
        'ADDPG(16,4)': 'Ant-v2_AQ_N2_O16_U4',
    }
    ant_m = {}
    ant_m.update(get_dataset(algo_path_Hopper_AQ_Over, base_path_Hopper_AQ_Over))


    base_path_Hopper_AQ_Over = '../data/Ant'
    algo_path_Hopper_AQ_Over = {
        'TD3': 'Ant-v2_TD3_N2',
        'ADDPG(4,1)': 'Ant-v2_AQ_N2_O4_U1',
        'ADDPG(4,2)': 'Ant-v2_AQ_N2_O4_U8',
        'ADDPG(4,4)': 'Ant-v2_AQ_N2_O4_U16',
        'ADDPG(4,8)': 'Ant-v2_AQ_N2_O4_U2',
        'ADDPG(4,16)': 'Ant-v2_AQ_N2_O1_U4',
    }
    ant_n = {}
    ant_n.update(get_dataset(algo_path_Hopper_AQ_Over, base_path_Hopper_AQ_Over))

    base_path_Hopper_AQ_Over = '../data/Ant'
    algo_path_Hopper_AQ_Over = {
        'DDPG': 'Ant-v2_ddpg_N1',
        'TD3': 'Ant-v2_TD3_N2',
        'ADDPG(4,4)': 'Ant-v2_AQ_N2_O4_U16',
        'AdaADDPG(1)': 'Ant-v2_SoftAQ_N2_T1',
    }
    ant_c = {}
    ant_c.update(get_dataset(algo_path_Hopper_AQ_Over, base_path_Hopper_AQ_Over))

    base_path_Hopper_AQ_Over = '../data/Ant'
    algo_path_Hopper_AQ_Over = {
        'DDPG': 'Ant-v2_ddpg_N1',
        'TD3': 'Ant-v2_TD3_N2',
        'AdaADDPG(1)': 'Ant-v2_SoftAQ_N2_T1',
        'AdaADDPG(2)': 'Ant-v2_SoftAQ_N2_T2',
        'AdaADDPG(5)': 'Ant-v2_SoftAQ_N2_T5',
        'AdaADDPG(10)': 'Ant-v2_SoftAQ_N2_T10',
        'AdaADDPG(100)': 'Ant-v2_SoftAQ_N2_T100',
    }
    ant_a = {}
    ant_a.update(get_dataset(algo_path_Hopper_AQ_Over, base_path_Hopper_AQ_Over))


    base_path_Ant_Compare = '../data/Walker2d'
    algo_path_Ant_Compare = {
        'DDPG': 'Walker2d-v2_ddpg_N1',
        'ADDPG(1,4)': 'Walker2d-v2_AQ_N2_O2_U4',
        'ADDPG(2,4)': 'Walker2d-v2_AQ_N2_O4_U4',
        'ADDPG(4,4)': 'Walker2d-v2_AQ_N2_O1_U4',
        'ADDPG(8,4)': 'Walker2d-v2_AQ_N2_O8_U4',
        'ADDPG(16,4)': 'Walker2d-v2_AQ_N2_O16_U4',
    }
    walk_m = {}
    walk_m.update(get_dataset(algo_path_Ant_Compare, base_path_Ant_Compare))

    base_path_Hopper = '../data/Walker2d'
    algo_path_Hopper = {
        'TD3': 'Walker2d-v2_TD3_N2',
        'ADDPG(4,1)': 'Walker2d-v2_AQ_N2_O4_U16',
        'ADDPG(4,2)': 'Walker2d-v2_AQ_N2_O4_U2',
        'ADDPG(4,4)': 'Walker2d-v2_AQ_N2_O1_U4',
        'ADDPG(4,8)': 'Walker2d-v2_AQ_N2_O4_U8',
        'ADDPG(4,16)': 'Walker2d-v2_AQ_N2_O4_U1',
    }
    walk_n = {}
    walk_n.update(get_dataset(algo_path_Hopper, base_path_Hopper))

    base_path_Ant = '../data/Walker2d'
    algo_path_Ant = {
        'DDPG': 'Walker2d-v2_ddpg_N1',
        'TD3': 'Walker2d-v2_TD3_N2',
        'ADDPG(4,4)': 'Walker2d-v2_AQ_N2_O1_U4',
        'AdaADDPG(1)': 'Walker2d-v2_SoftAQ_N2_T1',
    }
    walk_c = {}
    walk_c.update(get_dataset(algo_path_Ant, base_path_Ant))

    base_path_Walker2d = '../data/Walker2d'
    algo_path_Walker2d = {
        'DDPG': 'Walker2d-v2_ddpg_N1',
        'TD3': 'Walker2d-v2_TD3_N2',
        'AdaADDPG(1)': 'Walker2d-v2_SoftAQ_N2_T1',
        'AdaADDPG(2)': 'Walker2d-v2_SoftAQ_N2_T2',
        'AdaADDPG(5)': 'Walker2d-v2_SoftAQ_N2_T5',
        'AdaADDPG(10)': 'Walker2d-v2_SoftAQ_N2_T10',
        'AdaADDPG(100)': 'Walker2d-v2_SoftAQ_N2_T100',
    }
    walk_a = {}
    walk_a.update(get_dataset(algo_path_Walker2d, base_path_Walker2d))



    figure(Hopper_m, Hopper_n, Hopper_c, Hopper_a, ant_m, ant_n, ant_c, ant_a, walk_m, walk_n, walk_c, walk_a)
