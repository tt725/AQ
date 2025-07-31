import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import ast


def figure(roulette_m, roulette_n, roulette_c, roulette_a, reward3_m, reward3_n, reward3_c, reward3_a, reward4_m, reward4_n, reward4_c, reward4_a):
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
    ax1 = plt.axes(rect1)
    ax2 = plt.axes(rect2)
    ax4 = plt.axes(rect3)
    ax3 = plt.axes(rect4)
    ax5 = plt.axes(rect5)
    ax6 = plt.axes(rect6)
    ax8 = plt.axes(rect7)
    ax7 = plt.axes(rect8)
    ax9 = plt.axes(rect9)
    ax10 = plt.axes(rect10)
    ax12 = plt.axes(rect11)
    ax11 = plt.axes(rect12)

    label_m = [r'Q',
               r'AQ($1,4$)',
               r'AQ($2,4$)',
               r'AQ($4,4$)',
               r'AQ($8,4$)',
               r'AQ($16,4$)']
    x_value = [i for i in range(len(roulette_m[0]))]
    ax1.plot(x_value, roulette_m[0], linewidth=3.0, label=label_m[0], color='blue')
    ax1.plot(x_value, roulette_m[1], linewidth=3.0, label=label_m[1], color='green')
    ax1.plot(x_value, roulette_m[2], linewidth=3.0, label=label_m[2], color='c')
    ax1.plot(x_value, roulette_m[3], linewidth=3.0, label=label_m[3], color='m')
    ax1.plot(x_value, roulette_m[4], linewidth=3.0, label=label_m[4], color='y')
    ax1.plot(x_value, roulette_m[5], linewidth=3.0, label=label_m[5], color='#1f77b4')
    ax1.set_ylabel(r'$\Pr$[leave]', fontsize=15)
    ax1.set_xlabel(r'Steps (x10,000)', fontsize=15)
    ax1.set_xlim(0, len(roulette_m[0]))
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.4)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.grid()
    ax1.legend(fontsize=12, loc="upper left", handlelength=1)
    ax1.set_title(label=r'(a) AQ$(M,4)$ in Roulette', fontsize=15, y=-0.4)

    label_n = [r'DQ',
               r'AQ($4,1$)',
               r'AQ($4,2$)',
               r'AQ($4,4$)',
               r'AQ($4,8$)',
               r'AQ($4,16$)']
    x_value = [i for i in range(len(roulette_n[0]))]
    ax2.plot(x_value, roulette_n[0], linewidth=3.0, label=label_n[0], color='black')
    ax2.plot(x_value, roulette_n[1], linewidth=3.0, label=label_n[1], color='green')
    ax2.plot(x_value, roulette_n[2], linewidth=3.0, label=label_n[2], color='c')
    ax2.plot(x_value, roulette_n[3], linewidth=3.0, label=label_n[3], color='m')
    ax2.plot(x_value, roulette_n[4], linewidth=3.0, label=label_n[4], color='y')
    ax2.plot(x_value, roulette_n[5], linewidth=3.0, label=label_n[5], color='#1f77b4')
    ax2.set_ylabel(r'$\Pr$[leave]', fontsize=15)
    ax2.set_xlabel(r'Steps (x10,000)', fontsize=15)
    ax2.set_xlim(0, len(roulette_m[0]))
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax2.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.4)
    ax2.yaxis.set_major_locator(yy)
    ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax2.yaxis.get_offset_text().set_fontsize(15)
    ax2.grid()
    ax2.legend(fontsize=12, loc="upper left", handlelength=1)
    ax2.set_title(label=r'(b) AQ$(4,N)$ in Roulette', fontsize=15, y=-0.4)

    label_c = [r'Q',
               r'DQ',
               r'AQ($4,4$)',
               r'AdaAQ($1$)']
    x_value = [i for i in range(len(roulette_c[0]))]
    ax3.plot(x_value, roulette_c[0], linewidth=3.0, label=label_c[0], color='blue')
    ax3.plot(x_value, roulette_c[1], linewidth=3.0, label=label_c[1], color='black')
    ax3.plot(x_value, roulette_c[2], linewidth=3.0, label=label_c[2], color='m')
    ax3.plot(x_value, roulette_c[3], linewidth=3.0, label=label_c[3], color='red')
    ax3.set_ylabel(r'$\Pr$[leave]', fontsize=15)
    ax3.set_xlabel(r'Steps (x10,000)', fontsize=15)
    ax3.set_xlim(0, len(roulette_c[0]))
    ax3.set_ylim(0, 1.0)
    ax3.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax3.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.4)
    ax3.yaxis.set_major_locator(yy)
    ax3.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax3.yaxis.get_offset_text().set_fontsize(15)
    ax3.grid()
    ax3.legend(fontsize=12, loc="upper left", handlelength=1)
    ax3.set_title(label=r'(d) AdaAQ in Roulette', fontsize=15, y=-0.4)

    label_a = [r'Q',
               r'DQ',
               r'AdaAQ($1$)',
               r'AdaAQ($2$)',
               r'AdaAQ($5$)',
               r'AdaAQ($10$)',
               r'AdaAQ($100$)']
    x_value = [i for i in range(len(roulette_a[0]))]
    ax4.plot(x_value, roulette_a[0], linewidth=3.0, label=label_a[0], color='blue')
    ax4.plot(x_value, roulette_a[1], linewidth=3.0, label=label_a[1], color='black')
    ax4.plot(x_value, roulette_a[2], linewidth=3.0, label=label_a[2], color='red')
    ax4.plot(x_value, roulette_a[3], linewidth=3.0, label=label_a[3], color='green')
    ax4.plot(x_value, roulette_a[4], linewidth=3.0, label=label_a[4], color='c')
    ax4.plot(x_value, roulette_a[5], linewidth=3.0, label=label_a[5], color='m')
    ax4.plot(x_value, roulette_a[6], linewidth=3.0, label=label_a[6], color='y')
    ax4.set_ylabel(r'$\Pr$[leave]', fontsize=15)
    ax4.set_xlabel(r'Steps (x10,000)', fontsize=15)
    ax4.set_xlim(0, len(roulette_a[0]))
    ax4.set_ylim(0, 1.0)
    ax4.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax4.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.4)
    ax4.yaxis.set_major_locator(yy)
    ax4.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax4.yaxis.get_offset_text().set_fontsize(15)
    ax4.grid()
    ax4.legend(fontsize=12, loc="upper left", handlelength=1)
    ax4.set_title(label=r'(c) AdaAQ$(\tau)$ in Roulette', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(reward3_m[0]))]
    ax5.plot(x_value, reward3_m[0], linewidth=3.0, label=label_m[0], color='blue')
    ax5.plot(x_value, reward3_m[1], linewidth=3.0, label=label_m[1], color='green')
    ax5.plot(x_value, reward3_m[2], linewidth=3.0, label=label_m[2], color='c')
    ax5.plot(x_value, reward3_m[3], linewidth=3.0, label=label_m[3], color='m')
    ax5.plot(x_value, reward3_m[4], linewidth=3.0, label=label_m[4], color='y')
    ax5.plot(x_value, reward3_m[5], linewidth=3.0, label=label_m[5], color='#1f77b4')
    ax5.set_ylabel(r'Average reward per step', fontsize=15)
    ax5.set_xlabel(r'Steps (x500)', fontsize=15)
    ax5.set_xlim(0, len(reward3_m[0]))
    ax5.set_ylim(-1, -0.4)
    ax5.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax5.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.3)
    ax5.yaxis.set_major_locator(yy)
    ax5.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax5.yaxis.get_offset_text().set_fontsize(15)
    ax5.grid()
    ax5.legend(fontsize=12, loc="upper left", handlelength=1)
    ax5.set_title(label=r'(e) AQ$(M,4)$ in Grid 3x3', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(reward3_n[0]))]
    ax6.plot(x_value, reward3_n[0], linewidth=3.0, label=label_n[0], color='black')
    ax6.plot(x_value, reward3_n[1], linewidth=3.0, label=label_n[1], color='green')
    ax6.plot(x_value, reward3_n[2], linewidth=3.0, label=label_n[2], color='c')
    ax6.plot(x_value, reward3_n[3], linewidth=3.0, label=label_n[3], color='m')
    ax6.plot(x_value, reward3_n[4], linewidth=3.0, label=label_n[4], color='y')
    ax6.plot(x_value, reward3_n[5], linewidth=3.0, label=label_n[5], color='#1f77b4')
    ax6.set_ylabel(r'Average reward per step', fontsize=15)
    ax6.set_xlabel(r'Steps (x500)', fontsize=15)
    ax6.set_xlim(0, len(reward3_n[0]))
    ax6.set_ylim(-1, -0.4)
    ax6.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax6.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.3)
    ax6.yaxis.set_major_locator(yy)
    ax6.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax6.yaxis.get_offset_text().set_fontsize(15)
    ax6.grid()
    ax6.legend(fontsize=12, loc="upper left", handlelength=1)
    ax6.set_title(label=r'(f) AQ$(4,N)$ in Grid 3x3', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(reward3_c[0]))]
    ax7.plot(x_value, reward3_c[0], linewidth=3.0, label=label_c[0], color='blue')
    ax7.plot(x_value, reward3_c[1], linewidth=3.0, label=label_c[1], color='black')
    ax7.plot(x_value, reward3_c[2], linewidth=3.0, label=label_c[2], color='m')
    ax7.plot(x_value, reward3_c[3], linewidth=3.0, label=label_c[3], color='red')
    ax7.set_ylabel(r'Average reward per step', fontsize=15)
    ax7.set_xlabel(r'Steps (x500)', fontsize=15)
    ax7.set_xlim(0, len(reward3_c[0]))
    ax7.set_ylim(-1, -0.1)
    ax7.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax7.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.4)
    ax7.yaxis.set_major_locator(yy)
    ax7.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax7.yaxis.get_offset_text().set_fontsize(15)
    ax7.grid()
    ax7.legend(fontsize=12, loc="upper left", handlelength=1)
    ax7.set_title(label=r'(h) AdaAQ in Grid 3x3', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(reward3_a[0]))]
    ax8.plot(x_value, reward3_a[0], linewidth=3.0, label=label_a[0], color='blue')
    ax8.plot(x_value, reward3_a[1], linewidth=3.0, label=label_a[1], color='black')
    ax8.plot(x_value, reward3_a[2], linewidth=3.0, label=label_a[2], color='red')
    ax8.plot(x_value, reward3_a[3], linewidth=3.0, label=label_a[3], color='green')
    ax8.plot(x_value, reward3_a[4], linewidth=3.0, label=label_a[4], color='c')
    ax8.plot(x_value, reward3_a[5], linewidth=3.0, label=label_a[5], color='m')
    ax8.plot(x_value, reward3_a[6], linewidth=3.0, label=label_a[6], color='y')
    ax8.set_ylabel(r'Average reward per step', fontsize=15)
    ax8.set_xlabel(r'Steps (x500)', fontsize=15)
    ax8.set_xlim(0, len(reward3_a[0]))
    ax8.set_ylim(-1, -0.1)
    ax8.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax8.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.4)
    ax8.yaxis.set_major_locator(yy)
    ax8.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax8.yaxis.get_offset_text().set_fontsize(15)
    ax8.grid()
    ax8.legend(fontsize=12, loc="upper left", handlelength=1)
    ax8.set_title(label=r'(g) AdaAQ$(\tau)$ in Grid 3x3', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(reward4_m[0]))]
    ax9.plot(x_value, reward4_m[0], linewidth=3.0, label=label_m[0], color='blue')
    ax9.plot(x_value, reward4_m[1], linewidth=3.0, label=label_m[1], color='green')
    ax9.plot(x_value, reward4_m[2], linewidth=3.0, label=label_m[2], color='c')
    ax9.plot(x_value, reward4_m[3], linewidth=3.0, label=label_m[3], color='m')
    ax9.plot(x_value, reward4_m[4], linewidth=3.0, label=label_m[4], color='y')
    ax9.plot(x_value, reward4_m[5], linewidth=3.0, label=label_m[5], color='#1f77b4')
    ax9.set_ylabel(r'Average reward per step', fontsize=15)
    ax9.set_xlabel(r'Steps (x500)', fontsize=15)
    ax9.set_xlim(0, len(reward4_m[0]))
    ax9.set_ylim(-0.99, -0.71)
    ax9.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax9.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.1)
    ax9.yaxis.set_major_locator(yy)
    ax9.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax9.yaxis.get_offset_text().set_fontsize(15)
    ax9.grid()
    ax9.legend(fontsize=12, loc="upper left", handlelength=1)
    ax9.set_title(label=r'(i) AQ$(M,4)$ in Grid 4x4', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(reward4_n[0]))]
    ax10.plot(x_value, reward4_n[0], linewidth=3.0, label=label_n[0], color='black')
    ax10.plot(x_value, reward4_n[1], linewidth=3.0, label=label_n[1], color='green')
    ax10.plot(x_value, reward4_n[2], linewidth=3.0, label=label_n[2], color='c')
    ax10.plot(x_value, reward4_n[3], linewidth=3.0, label=label_n[3], color='m')
    ax10.plot(x_value, reward4_n[4], linewidth=3.0, label=label_n[4], color='y')
    ax10.plot(x_value, reward4_n[5], linewidth=3.0, label=label_n[5], color='#1f77b4')
    ax10.set_ylabel(r'Average reward per step', fontsize=15)
    ax10.set_xlabel(r'Steps (x500)', fontsize=15)
    ax10.set_xlim(0, len(reward4_n[0]))
    ax10.set_ylim(-0.99, -0.71)
    ax10.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax10.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.1)
    ax10.yaxis.set_major_locator(yy)
    ax10.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax10.yaxis.get_offset_text().set_fontsize(15)
    ax10.grid()
    ax10.legend(fontsize=12, loc="upper left", handlelength=1)
    ax10.set_title(label=r'(j) AQ$(4,N)$ in Grid 4x4', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(reward4_c[0]))]
    ax11.plot(x_value, reward4_c[0], linewidth=3.0, label=label_c[0], color='blue')
    ax11.plot(x_value, reward4_c[1], linewidth=3.0, label=label_c[1], color='black')
    ax11.plot(x_value, reward4_c[2], linewidth=3.0, label=label_c[2], color='m')
    ax11.plot(x_value, reward4_c[3], linewidth=3.0, label=label_c[3], color='red')
    ax11.set_ylabel(r'Average reward per step', fontsize=15)
    ax11.set_xlabel(r'Steps (x500)', fontsize=15)
    ax11.set_xlim(0, len(reward4_c[0]))
    ax11.set_ylim(-1, -0.45)
    ax11.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax11.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.3)
    ax11.yaxis.set_major_locator(yy)
    ax11.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax11.yaxis.get_offset_text().set_fontsize(15)
    ax11.grid()
    ax11.legend(fontsize=12, loc="upper left", handlelength=1)
    ax11.set_title(label=r'(l) AdaAQ in Grid 4x4', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(reward4_a[0]))]
    ax12.plot(x_value, reward4_a[0], linewidth=3.0, label=label_a[0], color='blue')
    ax12.plot(x_value, reward4_a[1], linewidth=3.0, label=label_a[1], color='black')
    ax12.plot(x_value, reward4_a[2], linewidth=3.0, label=label_a[2], color='red')
    ax12.plot(x_value, reward4_a[3], linewidth=3.0, label=label_a[3], color='green')
    ax12.plot(x_value, reward4_a[4], linewidth=3.0, label=label_a[4], color='c')
    ax12.plot(x_value, reward4_a[5], linewidth=3.0, label=label_a[5], color='m')
    ax12.plot(x_value, reward4_a[6], linewidth=3.0, label=label_a[6], color='y')
    ax12.set_ylabel(r'Average reward per step', fontsize=15)
    ax12.set_xlabel(r'Steps (x500)', fontsize=15)
    ax12.set_xlim(0, len(reward4_a[0]))
    ax12.set_ylim(-1, -0.45)
    ax12.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax12.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.3)
    ax12.yaxis.set_major_locator(yy)
    ax12.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax12.yaxis.get_offset_text().set_fontsize(15)
    ax12.grid()
    ax12.legend(fontsize=12, loc="upper left", handlelength=1)
    ax12.set_title(label=r'(k) AdaAQ$(\tau)$ in Grid 4x4', fontsize=15, y=-0.4)

    plt.savefig("./SuppleTableQ.pdf", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()


def get_P_value(dir):
    log = open(dir, 'r').readlines()
    A_left_P = log[-1][:]
    A_left_P = ast.literal_eval(A_left_P)
    return A_left_P


def get_reward_value(dir):
    log = open(dir, 'r').readlines()
    reward = log[-1][:]
    reward = ast.literal_eval(reward)
    return reward


if __name__ == "__main__":
    Q1_learning_dir = r'../Roulette/Result/log.Q 2025.03.16.13.39.50'
    y1_value = get_P_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Roulette/Result/log.AQ (1,4) 2025.03.17.14.43.04'
    y2_value = get_P_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Roulette/Result/log.AQ (2,4) 2025.03.17.14.43.11'
    y3_value = get_P_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Roulette/Result/log.AQ (4,4) 2025.03.17.14.43.20'
    y4_value = get_P_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Roulette/Result/log.AQ (8,4) 2025.03.17.14.43.34'
    y5_value = get_P_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Roulette/Result/log.AQ (16,4) 2025.03.17.14.43.41'
    y6_value = get_P_value(
        dir=Q6_learning_dir)
    roulette_m = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value]

    print("roulette_m:")
    for i in roulette_m:
        print(i[-1])

    Q1_learning_dir = r'../Roulette/Result/log.DQ 2025.03.16.13.40.04'
    y1_value = get_P_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Roulette/Result/log.AQ (4,1) 2025.03.16.13.40.19'
    y2_value = get_P_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Roulette/Result/log.AQ (4,2) 2025.03.16.23.45.49'
    y3_value = get_P_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Roulette/Result/log.AQ (4,4) 2025.03.17.14.43.20'
    y4_value = get_P_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Roulette/Result/log.AQ (4,8) 2025.03.19.10.24.39'
    y5_value = get_P_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Roulette/Result/log.AQ (4,16) 2025.03.17.14.43.47'
    y6_value = get_P_value(
        dir=Q6_learning_dir)
    roulette_n = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value]

    print("roulette_n:")
    for i in roulette_n:
        print(i[-1])

    Q1_learning_dir = r'../Roulette/Result/log.Q 2025.03.16.13.39.50'
    y1_value = get_P_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Roulette/Result/log.DQ 2025.03.16.13.40.04'
    y2_value = get_P_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Roulette/Result/log.AQ (4,4) 2025.03.17.14.43.20'
    y3_value = get_P_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Roulette/Result/log.SoftAQ (1) 2025.03.16.13.40.30'
    y4_value = get_P_value(
        dir=Q4_learning_dir)
    roulette_c = [y1_value, y2_value, y3_value, y4_value]

    print("roulette_c:")
    for i in roulette_c:
        print(i[-1])

    Q1_learning_dir = r'../Roulette/Result/log.Q 2025.03.16.13.39.50'
    y1_value = get_P_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Roulette/Result/log.DQ 2025.03.16.13.40.04'
    y2_value = get_P_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Roulette/Result/log.SoftAQ (1) 2025.03.16.13.40.30'
    y3_value = get_P_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Roulette/Result/log.SoftAQ (2) 2025.03.18.15.07.23'
    y4_value = get_P_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Roulette/Result/log.SoftAQ (5) 2025.03.18.15.07.28'
    y5_value = get_P_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Roulette/Result/log.SoftAQ (10) 2025.03.17.14.45.25'
    y6_value = get_P_value(
        dir=Q6_learning_dir)
    Q7_learning_dir = r'../Roulette/Result/log.SoftAQ (100) 2025.03.17.14.45.30'
    y7_value = get_P_value(
        dir=Q7_learning_dir)
    roulette_a = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value, y7_value]

    print("roulette_a:")
    for i in roulette_a:
        print(i[-1])

    Q1_learning_dir = r'../Gridworld3/Result/log.Q 2025.03.15.14.12.23'
    y1_value = get_reward_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Gridworld3/Result/log.AQ (1,4) 2025.03.15.17.22.55'
    y2_value = get_reward_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Gridworld3/Result/log.AQ (2,4) 2025.03.15.17.35.37'
    y3_value = get_reward_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Gridworld3/Result/log.AQ (4,4) 2025.03.15.16.42.34'
    y4_value = get_reward_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Gridworld3/Result/log.AQ (8,4) 2025.03.15.17.48.27'
    y5_value = get_reward_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Gridworld3/Result/log.AQ (16,4) 2025.03.15.18.01.19'
    y6_value = get_reward_value(
        dir=Q6_learning_dir)
    reward3_m = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value]

    print("reward3_m:")
    for i in reward3_m:
        print(i[-1])

    Q1_learning_dir = r'../Gridworld3/Result/log.DoubleQ 2025.03.15.14.15.53'
    y1_value = get_reward_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Gridworld3/Result/log.AQ (4,1) 2025.03.15.16.14.45'
    y2_value = get_reward_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Gridworld3/Result/log.AQ (4,2) 2025.03.15.16.28.46'
    y3_value = get_reward_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Gridworld3/Result/log.AQ (4,4) 2025.03.15.16.42.34'
    y4_value = get_reward_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Gridworld3/Result/log.AQ (4,8) 2025.03.15.16.56.12'
    y5_value = get_reward_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Gridworld3/Result/log.AQ (4,16) 2025.03.15.17.09.31'
    y6_value = get_reward_value(
        dir=Q6_learning_dir)
    reward3_n = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value]

    print("reward3_n:")
    for i in reward3_n:
        print(i[-1])

    Q1_learning_dir = r'../Gridworld3/Result/log.Q 2025.03.15.14.12.23'
    y1_value = get_reward_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Gridworld3/Result/log.DoubleQ 2025.03.15.14.15.53'
    y2_value = get_reward_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Gridworld3/Result/log.AQ (4,4) 2025.03.15.16.42.34'
    y3_value = get_reward_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Gridworld3/Result/log.SoftAQ (1) 2025.03.15.14.21.18'
    y4_value = get_reward_value(
        dir=Q4_learning_dir)
    reward3_c = [y1_value, y2_value, y3_value, y4_value]

    print("reward3_c:")
    for i in reward3_c:
        print(i[-1])

    Q1_learning_dir = r'../Gridworld3/Result/log.Q 2025.03.15.14.12.23'
    y1_value = get_reward_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Gridworld3/Result/log.DoubleQ 2025.03.15.14.15.53'
    y2_value = get_reward_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Gridworld3/Result/log.SoftAQ (1) 2025.03.15.14.21.18'
    y3_value = get_reward_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Gridworld3/Result/log.SoftAQ (2) 2025.03.15.14.48.18'
    y4_value = get_reward_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Gridworld3/Result/log.SoftAQ (5) 2025.03.15.15.15.14'
    y5_value = get_reward_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Gridworld3/Result/log.SoftAQ (10) 2025.03.15.15.41.57'
    y6_value = get_reward_value(
        dir=Q6_learning_dir)
    Q7_learning_dir = r'../Gridworld3/Result/log.SoftAQ (100) 2025.03.15.16.09.38'
    y7_value = get_reward_value(
        dir=Q7_learning_dir)
    reward3_a = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value, y7_value]

    print("reward3_a:")
    for i in reward3_a:
        print(i[-1])

    Q1_learning_dir = r'../Gridworld4/Result/log.Q 2025.03.16.11.44.21'
    y1_value = get_reward_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Gridworld4/Result/log.AQ (1,4) 2025.03.16.00.02.53'
    y2_value = get_reward_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Gridworld4/Result/log.AQ (2,4) 2025.03.16.00.17.23'
    y3_value = get_reward_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Gridworld4/Result/log.AQ (4,4) 2025.03.15.23.17.48'
    y4_value = get_reward_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Gridworld4/Result/log.AQ (8,4) 2025.03.16.00.31.45'
    y5_value = get_reward_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Gridworld4/Result/log.AQ (16,4) 2025.03.16.00.46.11'
    y6_value = get_reward_value(
        dir=Q6_learning_dir)
    reward4_m = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value]

    print("reward4_m:")
    for i in reward4_m:
        print(i[-1])

    Q1_learning_dir = r'../Gridworld4/Result/log.DoubleQ 2025.03.16.11.44.15'
    y1_value = get_reward_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Gridworld4/Result/log.AQ (4,1) 2025.03.15.22.43.02'
    y2_value = get_reward_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Gridworld4/Result/log.AQ (4,2) 2025.03.15.23.01.15'
    y3_value = get_reward_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Gridworld4/Result/log.AQ (4,4) 2025.03.15.23.17.48'
    y4_value = get_reward_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Gridworld4/Result/log.AQ (4,8) 2025.03.15.23.33.09'
    y5_value = get_reward_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Gridworld4/Result/log.AQ (4,16) 2025.03.15.23.48.12'
    y6_value = get_reward_value(
        dir=Q6_learning_dir)
    reward4_n = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value]

    print("reward4_n:")
    for i in reward4_n:
        print(i[-1])

    Q1_learning_dir = r'../Gridworld4/Result/log.Q 2025.03.16.11.44.21'
    y1_value = get_reward_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Gridworld4/Result/log.DoubleQ 2025.03.16.11.44.15'
    y2_value = get_reward_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Gridworld4/Result/log.AQ (4,4) 2025.03.15.23.17.48'
    y3_value = get_reward_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Gridworld4/Result/log.SoftAQ (1) 2025.03.15.22.43.09'
    y4_value = get_reward_value(
        dir=Q4_learning_dir)
    reward4_c = [y1_value, y2_value, y3_value, y4_value]

    print("reward4_c:")
    for i in reward4_c:
        print(i[-1])

    Q1_learning_dir = r'../Gridworld4/Result/log.Q 2025.03.16.11.44.21'
    y1_value = get_reward_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Gridworld4/Result/log.DoubleQ 2025.03.16.11.44.15'
    y2_value = get_reward_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Gridworld4/Result/log.SoftAQ (1) 2025.03.15.22.43.09'
    y3_value = get_reward_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Gridworld4/Result/log.SoftAQ (2) 2025.03.15.23.19.31'
    y4_value = get_reward_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Gridworld4/Result/log.SoftAQ (5) 2025.03.15.23.51.39'
    y5_value = get_reward_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Gridworld4/Result/log.SoftAQ (10) 2025.03.16.00.22.18'
    y6_value = get_reward_value(
        dir=Q6_learning_dir)
    Q7_learning_dir = r'../Gridworld4/Result/log.SoftAQ (100) 2025.03.16.00.53.02'
    y7_value = get_reward_value(
        dir=Q7_learning_dir)
    reward4_a = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value, y7_value]

    print("reward4_a:")
    for i in reward4_a:
        print(i[-1])
    
    figure(roulette_m, roulette_n, roulette_c, roulette_a, reward3_m, reward3_n, reward3_c, reward3_a, reward4_m, reward4_n, reward4_c, reward4_a)

