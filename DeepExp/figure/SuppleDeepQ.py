from utils.plotter import Plotter
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def get_date(env_name, x_label, y_label, indexList, runs):
    plotter = Plotter(env_name=env_name, merged=True, x_label=x_label, y_label=y_label,
                      ci="se", EMA=True, runs=runs)
    data = plotter.result_indexList(indexList, mode='Train')
    return data


def figure(copter_m,copter_n,copter_c,copter_a,breakout_m, breakout_n, breakout_c, breakout_a, asterix_m, asterix_n, asterix_c, asterix_a):
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

    label_m = [r'DQN',
               r'ADQN(1,4)',
               r'ADQN(2,4)',
               r'ADQN(4,4)',
               r'ADQN(8,4)',
               r'ADQN(16,4)']

    color_m = ['blue',
                'green',
                'c',
                'm',
                'y',
               '#1f77b4']

    print("breakout_m")
    for i in range(len(breakout_m)):
        ys = []
        for result in breakout_m[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = breakout_m[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax1.plot(x_mean, y_mean, linewidth=1.5, label=label_m[i], color=color_m[i])
        ax1.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax1.set_ylabel("Average score per episode", fontsize=15)
    ax1.set_xlabel("Frames", fontsize=15)
    ax1.set_xlim(0, 1e6)
    ax1.set_ylim(0, 15)
    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(5)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.xaxis.set_major_locator(xx)
    ax1.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax1.xaxis.get_offset_text().set_fontsize(15)
    ax1.set_title(label=r'(e) ADQN$(M,4)$ in Breakout', fontsize=15, y=-0.4)
    ax1.grid()
    ax1.legend(fontsize=12, loc="upper left", handlelength=1, frameon=True, framealpha=0.8)

    label_n = [r'DDQN',
               r'ADQN(4,1)',
               r'ADQN(4,2)',
               r'ADQN(4,4)',
               r'ADQN(4,8)',
               r'ADQN(4,16)']

    color_n = ['black',
               'green',
               'c',
               'm',
               'y',
               '#1f77b4']

    print("breakout_n")
    for i in range(len(breakout_n)):
        ys = []
        for result in breakout_n[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = breakout_n[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax2.plot(x_mean, y_mean, linewidth=1.5, label=label_n[i], color=color_n[i])
        ax2.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax2.set_ylabel("Average score per episode", fontsize=15)
    ax2.set_xlabel("Frames", fontsize=15)
    ax2.set_xlim(0, 1e6)
    ax2.set_ylim(0, 15)
    ax2.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax2.xaxis.set_major_locator(xx)
    yy = MultipleLocator(5)
    ax2.yaxis.set_major_locator(yy)
    ax2.yaxis.get_offset_text().set_fontsize(15)
    ax2.xaxis.set_major_locator(xx)
    ax2.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax2.xaxis.get_offset_text().set_fontsize(15)
    ax2.set_title(label=r'(f) ADQN$(4,N)$ in Breakout', fontsize=15, y=-0.4)
    ax2.grid()
    ax2.legend(fontsize=12, loc="upper left", handlelength=1, frameon=True, framealpha=0.8)

    label_c = [r'DQN',
               r'DDQN',
               r'ADQN(4,4)',
               r'AdaADQN(1)']

    color_c = ['blue',
               'black',
               'm',
               'red']

    print("breakout_c")
    for i in range(len(breakout_c)):
        ys = []
        for result in breakout_c[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = breakout_c[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax3.plot(x_mean, y_mean, linewidth=1.5, label=label_c[i], color=color_c[i])
        ax3.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax3.set_ylabel("Average score per episode", fontsize=15)
    ax3.set_xlabel("Frames", fontsize=15)
    ax3.set_xlim(0, 1e6)
    ax3.set_ylim(0, 15)
    ax3.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax3.xaxis.set_major_locator(xx)
    yy = MultipleLocator(5)
    ax3.yaxis.set_major_locator(yy)
    ax3.yaxis.get_offset_text().set_fontsize(15)
    ax3.xaxis.set_major_locator(xx)
    ax3.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax3.xaxis.get_offset_text().set_fontsize(15)
    ax3.set_title(label=r'(h) AdaADQN in Breakout', fontsize=15, y=-0.4)
    ax3.grid()
    ax3.legend(fontsize=12, loc="upper left", handlelength=1, frameon=True, framealpha=0.8)

    label_a = [r'DQN',
                 r"DDQN",
                 r'AdaADQN(1)',
                 r'AdaADQN(2)',
                 r'AdaADQN(5)',
                 r'AdaADQN(10)',
                 r'AdaADQN(100)']

    color_a = ['blue',
                 'black',
                 'red',
                 'green',
                 'c',
                 'm',
                 'y']
    print("breakout_a")
    for i in range(len(breakout_a)):
        ys = []
        for result in breakout_a[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = breakout_a[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax4.plot(x_mean, y_mean, linewidth=1.5, label=label_a[i], color=color_a[i])
        ax4.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax4.set_ylabel("Average score per episode", fontsize=15)
    ax4.set_xlabel("Frames", fontsize=15)
    ax4.set_xlim(0, 1e6)
    ax4.set_ylim(0, 15)
    ax4.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    yy = MultipleLocator(5)
    ax4.yaxis.set_major_locator(yy)
    ax4.yaxis.get_offset_text().set_fontsize(15)
    ax4.xaxis.set_major_locator(xx)
    ax4.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax4.xaxis.get_offset_text().set_fontsize(15)
    ax4.grid()
    ax4.legend(fontsize=12, loc="upper left", handlelength=1, frameon=True, framealpha=0.8)
    ax4.set_title(label=r'(g) AdaADQN$(\tau)$ in Breakout', fontsize=15, y=-0.4)


    print("asterix_m")
    for i in range(len(asterix_m)):
        ys = []
        for result in asterix_m[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = asterix_m[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax5.plot(x_mean, y_mean, linewidth=1.5, label=label_m[i], color=color_m[i])
        ax5.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax5.set_ylabel("Average score per episode", fontsize=15)
    ax5.set_xlabel("Frames", fontsize=15)
    ax5.set_xlim(0, 3e6)
    ax5.set_ylim(0, 18)
    ax5.tick_params(labelsize=15)
    xx = MultipleLocator(1e6)
    yy = MultipleLocator(5)
    ax5.yaxis.set_major_locator(yy)
    ax5.yaxis.get_offset_text().set_fontsize(15)
    ax5.xaxis.set_major_locator(xx)
    ax5.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax5.xaxis.get_offset_text().set_fontsize(15)
    ax5.grid()
    ax5.legend(fontsize=12, loc="upper left", handlelength=1, frameon=True, framealpha=0.8)
    ax5.set_title(label=r'(i) ADQN$(M,4)$ in Asterix', fontsize=15, y=-0.4)


    print("asterix_n")
    for i in range(len(asterix_n)):
        ys = []
        for result in asterix_n[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = asterix_n[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax6.plot(x_mean, y_mean, linewidth=1.5, label=label_n[i], color=color_n[i])
        ax6.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax6.set_ylabel("Average score per episode", fontsize=15)
    ax6.set_xlabel("Frames", fontsize=15)
    ax6.set_xlim(0, 3e6)
    ax6.set_ylim(0, 18)
    ax6.tick_params(labelsize=15)
    xx = MultipleLocator(1e6)
    yy = MultipleLocator(5)
    ax6.yaxis.set_major_locator(yy)
    ax6.yaxis.get_offset_text().set_fontsize(15)
    ax6.xaxis.set_major_locator(xx)
    ax6.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax6.xaxis.get_offset_text().set_fontsize(15)
    ax6.grid()
    ax6.legend(fontsize=12, loc="upper left", handlelength=1, frameon=True, framealpha=0.8)
    ax6.set_title(label=r'(j) ADQN$(4,N)$ in Asterix', fontsize=15, y=-0.4)

    print("asterix_c")
    for i in range(len(asterix_c)):
        ys = []
        for result in asterix_c[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = asterix_c[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax7.plot(x_mean, y_mean, linewidth=1.5, label=label_c[i], color=color_c[i])
        ax7.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax7.set_ylabel("Average score per episode", fontsize=15)
    ax7.set_xlabel("Frames", fontsize=15)
    ax7.set_xlim(0, 3e6)
    ax7.set_ylim(0, 18)
    ax7.tick_params(labelsize=15)
    xx = MultipleLocator(1e6)
    yy = MultipleLocator(5)
    ax7.yaxis.set_major_locator(yy)
    ax7.yaxis.get_offset_text().set_fontsize(15)
    ax7.xaxis.set_major_locator(xx)
    ax7.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax7.xaxis.get_offset_text().set_fontsize(15)
    ax7.grid()
    ax7.legend(fontsize=12, loc="upper left", handlelength=1, frameon=True, framealpha=0.8)
    ax7.set_title(label=r'(l) AdaADQN in Asterix', fontsize=15, y=-0.4)

    print("asterix_a")
    for i in range(len(asterix_a)):
        ys = []
        for result in asterix_a[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = asterix_a[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax8.plot(x_mean, y_mean, linewidth=1.5, label=label_a[i], color=color_a[i])
        ax8.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax8.set_ylabel("Average score per episode", fontsize=15)
    ax8.set_xlabel("Frames", fontsize=15)
    ax8.set_xlim(0, 3e6)
    ax8.set_ylim(0, 18)
    ax8.tick_params(labelsize=15)
    xx = MultipleLocator(1e6)
    yy = MultipleLocator(5)
    ax8.yaxis.set_major_locator(yy)
    ax8.yaxis.get_offset_text().set_fontsize(15)
    ax8.xaxis.set_major_locator(xx)
    ax8.xaxis.get_major_formatter().set_powerlimits((6, 6))
    ax8.xaxis.get_offset_text().set_fontsize(15)
    ax8.grid()
    ax8.legend(fontsize=12, loc="upper left", handlelength=1, frameon=True, framealpha=0.8)
    ax8.set_title(label=r'(k) AdaADQN$(\tau)$ in Asterix', fontsize=15, y=-0.4)


    print("copter_m")
    for i in range(len(copter_m)):
        ys = []
        for result in copter_m[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = copter_m[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax9.plot(x_mean, y_mean, linewidth=1.5, label=label_m[i], color=color_m[i])
        ax9.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax9.set_ylabel("Average score per episode", fontsize=15)
    ax9.set_xlabel("Frames", fontsize=15)
    ax9.set_xlim(0, 1e6)
    ax9.set_ylim(0, 40)
    ax9.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax9.xaxis.set_major_locator(xx)
    yy = MultipleLocator(12)
    ax9.yaxis.set_major_locator(yy)
    ax9.yaxis.get_offset_text().set_fontsize(15)
    ax9.xaxis.set_major_locator(xx)
    ax9.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax9.xaxis.get_offset_text().set_fontsize(15)
    ax9.set_title(label=r'(a) ADQN$(M,4)$ in Pixelcopter', fontsize=15, y=-0.4)
    ax9.grid()
    ax9.legend(fontsize=12, loc="upper left", handlelength=1, frameon=True, framealpha=0.8)


    print("copter_n")
    for i in range(len(copter_n)):
        ys = []
        for result in copter_n[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = copter_n[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax10.plot(x_mean, y_mean, linewidth=1.5, label=label_n[i], color=color_n[i])
        ax10.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax10.set_ylabel("Average score per episode", fontsize=15)
    ax10.set_xlabel("Frames", fontsize=15)
    ax10.set_xlim(0, 1e6)
    ax10.set_ylim(0, 40)
    ax10.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax10.xaxis.set_major_locator(xx)
    yy = MultipleLocator(12)
    ax10.yaxis.set_major_locator(yy)
    ax10.yaxis.get_offset_text().set_fontsize(15)
    ax10.xaxis.set_major_locator(xx)
    ax10.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax10.xaxis.get_offset_text().set_fontsize(15)
    ax10.set_title(label=r'(b) ADQN$(4,N)$ in Pixelcopter', fontsize=15, y=-0.4)
    ax10.grid()
    ax10.legend(fontsize=12, loc="upper left", handlelength=1, frameon=True, framealpha=0.8)


    print("copter_c")
    for i in range(len(copter_c)):
        ys = []
        for result in copter_c[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = copter_c[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax12.plot(x_mean, y_mean, linewidth=1.5, label=label_c[i], color=color_c[i])
        ax12.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax12.set_ylabel("Average score per episode", fontsize=15)
    ax12.set_xlabel("Frames", fontsize=15)
    ax12.set_xlim(0, 1e6)
    ax12.set_ylim(0, 40)
    ax12.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax12.xaxis.set_major_locator(xx)
    yy = MultipleLocator(12)
    ax12.yaxis.set_major_locator(yy)
    ax12.yaxis.get_offset_text().set_fontsize(15)
    ax12.xaxis.set_major_locator(xx)
    ax12.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax12.xaxis.get_offset_text().set_fontsize(15)
    ax12.set_title(label=r'(d) AdaADQN in Pixelcopter', fontsize=15, y=-0.4)
    ax12.grid()
    ax12.legend(fontsize=12, loc="upper left", handlelength=1, frameon=True, framealpha=0.8)

    print("copter a")
    for i in range(len(copter_a)):
        ys = []
        for result in copter_a[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = copter_a[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax11.plot(x_mean, y_mean, linewidth=1.5, label=label_a[i], color=color_a[i])
        ax11.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax11.set_ylabel("Average score per episode", fontsize=15)
    ax11.set_xlabel("Frames", fontsize=15)
    ax11.set_xlim(0, 1e6)
    ax11.set_ylim(0, 40)
    ax11.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    yy = MultipleLocator(12)
    ax11.yaxis.set_major_locator(yy)
    ax11.yaxis.get_offset_text().set_fontsize(15)
    ax11.xaxis.set_major_locator(xx)
    ax11.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax11.xaxis.get_offset_text().set_fontsize(15)
    ax11.grid()
    ax11.legend(fontsize=12, loc="upper left", handlelength=1, frameon=True, framealpha=0.8)
    ax11.set_title(label=r'(c) AdaADQN$(\tau)$ in Pixelcopter', fontsize=15, y=-0.4)


    plt.savefig("./SuppleDeepQ.pdf", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()


if __name__ == "__main__":
    x_label = 'Step'
    y_label = 'Average Return'
    copter_m = get_date(env_name='copter', x_label=x_label, y_label=y_label,
                        indexList=[1,11,12,13,14,15],
                        runs=20)
    copter_n = get_date(env_name='copter', x_label=x_label, y_label=y_label,
                        indexList=[2,16,17,13,18,19],
                        runs=20)
    copter_c = get_date(env_name='copter', x_label=x_label, y_label=y_label,
                        indexList=[1, 2, 13, 20],
                        runs=20)
    copter_a = get_date(env_name='copter', x_label=x_label, y_label=y_label,
                        indexList=[1, 2, 20, 21, 22, 23, 24],
                        runs=20)
    breakout_m = get_date(env_name='minatar_Breakout', x_label=x_label, y_label=y_label,
                          indexList=[1,11,12,13,14,15],
                          runs=20)
    breakout_n = get_date(env_name='minatar_Breakout', x_label=x_label, y_label=y_label,
                          indexList=[2,16,17,13,18,19],
                          runs=20)
    breakout_c = get_date(env_name='minatar_Breakout', x_label=x_label, y_label=y_label,
                          indexList=[1, 2, 13, 20],
                          runs=20)
    breakout_a = get_date(env_name='minatar_Breakout', x_label=x_label, y_label=y_label,
                             indexList=[1, 2, 20, 21, 22, 23, 24],
                             runs=20)
    asterix_m = get_date(env_name='minatar_Asterix', x_label=x_label, y_label=y_label,
                                indexList=[1,11,12,13,14,15],
                                runs=20)
    asterix_n = get_date(env_name='minatar_Asterix', x_label=x_label, y_label=y_label,
                                indexList=[2,16,17,13,18,19],
                                runs=20)
    asterix_c = get_date(env_name='minatar_Asterix', x_label=x_label, y_label=y_label,
                                 indexList=[1, 2, 13, 20],
                                 runs=20)
    asterix_a = get_date(env_name='minatar_Asterix', x_label=x_label, y_label=y_label,
                               indexList=[1, 2, 21, 20, 22, 23, 24],
                               runs=20)
    figure(copter_m,copter_n,copter_c,copter_a,breakout_m, breakout_n, breakout_c, breakout_a, asterix_m, asterix_n, asterix_c, asterix_a)
