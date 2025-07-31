import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import ast


def figure(multiarm_m, multiarm_n, multiarm_c, multiarm_a, multiarm_sota, policy_sota, reward3_sota, reward4_sota):
    fig = plt.figure(figsize=(13, 7))
    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect1 = [0.05, 0.65, 0.18, 0.34]
    rect2 = [0.3, 0.65, 0.18, 0.34]
    rect4 = [0.55, 0.65, 0.18, 0.34]
    rect3 = [0.8, 0.65, 0.18, 0.34]
    rect5 = [0.05, 0.13, 0.18, 0.34]
    rect6 = [0.3, 0.13, 0.18, 0.34]
    rect7 = [0.55, 0.13, 0.18, 0.34]
    rect8 = [0.8, 0.13, 0.18, 0.34]
    ax1 = plt.axes(rect1)
    ax2 = plt.axes(rect2)
    ax3 = plt.axes(rect3)
    ax4 = plt.axes(rect4)
    ax5 = plt.axes(rect5)
    ax6 = plt.axes(rect6)
    ax7 = plt.axes(rect7)
    ax8 = plt.axes(rect8)




    label_m = [r'Q',
               r'AQ($1,4$)',
               r'AQ($2,4$)',
               r'AQ($4,4$)',
               r'AQ($8,4$)',
               r'AQ($16,4$)']
    x_value = [i for i in range(len(multiarm_m[0]))]
    ax1.plot(x_value, multiarm_m[0], linewidth=3.0, label=label_m[0], color='blue')
    ax1.plot(x_value, multiarm_m[1], linewidth=3.0, label=label_m[1], color='green')
    ax1.plot(x_value, multiarm_m[2], linewidth=3.0, label=label_m[2], color='c')
    ax1.plot(x_value, multiarm_m[3], linewidth=3.0, label=label_m[3], color='m')
    ax1.plot(x_value, multiarm_m[4], linewidth=3.0, label=label_m[4], color='y')
    ax1.plot(x_value, multiarm_m[5], linewidth=3.0, label=label_m[5], color='#1f77b4')
    ax1.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax1.set_xlabel(r'Steps (x100)', fontsize=15)
    ax1.set_xlim(0, len(multiarm_m[0]))
    ax1.set_ylim(-2, 4)
    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.grid()
    ax1.legend(fontsize=12, loc="upper left", handlelength=1)
    ax1.set_title(label=r'(a) AQ with diff $M$', fontsize=15, y=-0.4)



    label_n = [r'DQ',
               r'AQ($4,1$)',
               r'AQ($4,2$)',
               r'AQ($4,4$)',
               r'AQ($4,8$)',
               r'AQ($4,16$)']
    x_value = [i for i in range(len(multiarm_n[0]))]
    ax2.plot(x_value, multiarm_n[0], linewidth=3.0, label=label_n[0], color='black')
    ax2.plot(x_value, multiarm_n[1], linewidth=3.0, label=label_n[1], color='green')
    ax2.plot(x_value, multiarm_n[2], linewidth=3.0, label=label_n[2], color='c')
    ax2.plot(x_value, multiarm_n[3], linewidth=3.0, label=label_n[3], color='m')
    ax2.plot(x_value, multiarm_n[4], linewidth=3.0, label=label_n[4], color='y')
    ax2.plot(x_value, multiarm_n[5], linewidth=3.0, label=label_n[5], color='#1f77b4')
    ax2.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax2.set_xlabel(r'Steps (x100)', fontsize=15)
    ax2.set_xlim(0, len(multiarm_m[0]))
    ax2.set_ylim(-3, 3)
    ax2.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax2.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2)
    ax2.yaxis.set_major_locator(yy)
    ax2.yaxis.get_offset_text().set_fontsize(15)
    ax2.grid()
    ax2.legend(fontsize=12, loc="upper left", handlelength=1)
    ax2.set_title(label=r'(b) AQ with diff $N$', fontsize=15, y=-0.4)




    label_c = [r'Q',
               r'DQ',
               r'AQ($4,8$)',
               r'AdaAQ($1$)']
    x_value = [i for i in range(len(multiarm_c[0]))]
    ax3.plot(x_value, multiarm_c[0], linewidth=3.0, label=label_c[0], color='blue')
    ax3.plot(x_value, multiarm_c[1], linewidth=3.0, label=label_c[1], color='black')
    ax3.plot(x_value, multiarm_c[2], linewidth=3.0, label=label_c[2], color='y')
    ax3.plot(x_value, multiarm_c[3], linewidth=3.0, label=label_c[3], color='red')
    ax3.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax3.set_xlabel(r'Steps (x100)', fontsize=15)
    ax3.set_xlim(0, len(multiarm_c[0]))
    ax3.set_ylim(-3.0, 4)
    ax3.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax3.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2)
    ax3.yaxis.set_major_locator(yy)
    ax3.yaxis.get_offset_text().set_fontsize(15)
    ax3.grid()
    ax3.legend(fontsize=12, loc="upper left", handlelength=1)
    ax3.set_title(label=r'(d) AdaAQ VS AQ', fontsize=15, y=-0.4)



    label_a = [r'Q',
               r'DQ',
               r'AdaAQ($1$)',
               r'AdaAQ($2$)',
               r'AdaAQ($5$)',
               r'AdaAQ($10$)',
               r'AdaAQ($100$)']
    x_value = [i for i in range(len(multiarm_a[0]))]
    ax4.plot(x_value, multiarm_a[0], linewidth=3.0, label=label_a[0], color='blue')
    ax4.plot(x_value, multiarm_a[1], linewidth=3.0, label=label_a[1], color='black')
    ax4.plot(x_value, multiarm_a[2], linewidth=3.0, label=label_a[2], color='red')
    ax4.plot(x_value, multiarm_a[3], linewidth=3.0, label=label_a[3], color='green')
    ax4.plot(x_value, multiarm_a[4], linewidth=3.0, label=label_a[4], color='c')
    ax4.plot(x_value, multiarm_a[5], linewidth=3.0, label=label_a[5], color='m')
    ax4.plot(x_value, multiarm_a[6], linewidth=3.0, label=label_a[6], color='y')
    ax4.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax4.set_xlabel(r'Steps (x100)', fontsize=15)
    ax4.set_xlim(0, len(multiarm_c[0]))
    ax4.set_ylim(-3, 4)
    ax4.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax4.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2)
    ax4.yaxis.set_major_locator(yy)
    ax4.yaxis.get_offset_text().set_fontsize(15)
    ax4.grid()
    ax4.legend(fontsize=12, loc="upper left", handlelength=1)
    ax4.set_title(label=r'(c) AdaAQ with diff $\tau$', fontsize=15, y=-0.4)


    label_sota = [r'AvgQ',
             r"MQ",
             r'SCQ',
             r'SoftQ',
             r'WDQ',
             r'REDQ',
             r'EBQL',
             r"AdaEQ",
             r'AdaAQ']

    x_value = [i for i in range(len(multiarm_sota[0]))]
    ax5.plot(x_value, multiarm_sota[0], linewidth=3.0, label=label_sota[0], color='blue')
    ax5.plot(x_value, multiarm_sota[1], linewidth=3.0, label=label_sota[1], color='black')
    ax5.plot(x_value, multiarm_sota[2], linewidth=3.0, label=label_sota[2], color='green')
    ax5.plot(x_value, multiarm_sota[3], linewidth=3.0, label=label_sota[3], color='c')
    ax5.plot(x_value, multiarm_sota[4], linewidth=3.0, label=label_sota[4], color='m')
    ax5.plot(x_value, multiarm_sota[5], linewidth=3.0, label=label_sota[5], color='y')
    ax5.plot(x_value, multiarm_sota[6], linewidth=3.0, label=label_sota[6], color='#1f77b4')
    ax5.plot(x_value, multiarm_sota[7], linewidth=3.0, label=label_sota[7], color='#ff7f0e')
    ax5.plot(x_value, multiarm_sota[8], linewidth=3.0, label=label_sota[8], color='red')
    ax5.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax5.set_xlabel(r'Steps (x100)', fontsize=15)
    ax5.set_xlim(0, len(multiarm_sota[0]))
    ax5.set_ylim(-2.5, 2.0)
    ax5.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax5.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2)
    ax5.yaxis.set_major_locator(yy)
    ax5.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax5.yaxis.get_offset_text().set_fontsize(15)
    ax5.grid()
    ax5.legend(fontsize=12, loc="upper left", handlelength=1)
    ax5.set_title(label=r'(e) SOTA in Multi-armed', fontsize=15, y=-0.4)



    x_value = [i for i in range(len(policy_sota[0]))]
    ax6.plot(x_value, policy_sota[0], linewidth=3.0, label=label_sota[0], color='blue')
    ax6.plot(x_value, policy_sota[1], linewidth=3.0, label=label_sota[1], color='black')
    ax6.plot(x_value, policy_sota[2], linewidth=3.0, label=label_sota[2], color='green')
    ax6.plot(x_value, policy_sota[3], linewidth=3.0, label=label_sota[3], color='c')
    ax6.plot(x_value, policy_sota[4], linewidth=3.0, label=label_sota[4], color='m')
    ax6.plot(x_value, policy_sota[5], linewidth=3.0, label=label_sota[5], color='y')
    ax6.plot(x_value, policy_sota[6], linewidth=3.0, label=label_sota[6], color='#1f77b4')
    ax6.plot(x_value, policy_sota[7], linewidth=3.0, label=label_sota[7], color='#ff7f0e')
    ax6.plot(x_value, policy_sota[8], linewidth=3.0, label=label_sota[8], color='red')
    ax6.set_ylabel(r'$\Pr$[leave]', fontsize=15)
    ax6.set_xlabel(r'Steps (x10,000)', fontsize=15)
    ax6.set_xlim(0, len(policy_sota[0]))
    ax6.set_ylim(0, 1.0)
    ax6.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax6.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.4)
    ax6.yaxis.set_major_locator(yy)
    ax6.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax6.yaxis.get_offset_text().set_fontsize(15)
    ax6.grid()
    ax6.legend(fontsize=12, loc="upper left", handlelength=1)
    ax6.set_title(label=r'(f) SOTA in Roulette', fontsize=15, y=-0.4)



    x_value = [i for i in range(len(reward3_sota[0]))]
    ax7.plot(x_value, reward3_sota[0], linewidth=3.0, label=label_sota[0], color='blue')
    ax7.plot(x_value, reward3_sota[1], linewidth=3.0, label=label_sota[1], color='black')
    ax7.plot(x_value, reward3_sota[2], linewidth=3.0, label=label_sota[2], color='green')
    ax7.plot(x_value, reward3_sota[3], linewidth=3.0, label=label_sota[3], color='c')
    ax7.plot(x_value, reward3_sota[4], linewidth=3.0, label=label_sota[4], color='m')
    ax7.plot(x_value, reward3_sota[5], linewidth=3.0, label=label_sota[5], color='y')
    ax7.plot(x_value, reward3_sota[6], linewidth=3.0, label=label_sota[6], color='#1f77b4')
    ax7.plot(x_value, reward3_sota[7], linewidth=3.0, label=label_sota[7], color='#ff7f0e')
    ax7.plot(x_value, reward3_sota[8], linewidth=3.0, label=label_sota[8], color='red')
    ax7.set_ylabel(r'Average reward per step', fontsize=15)
    ax7.set_xlabel(r'Steps (x500)', fontsize=15)
    ax7.set_xlim(0, len(reward3_sota[0]))
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
    ax7.set_title(label=r'(g) SOTA in Gridworld 3x3', fontsize=15, y=-0.4)



    x_value = [i for i in range(len(reward4_sota[0]))]
    ax8.plot(x_value, reward4_sota[0], linewidth=3.0, label=label_sota[0], color='blue')
    ax8.plot(x_value, reward4_sota[1], linewidth=3.0, label=label_sota[1], color='black')
    ax8.plot(x_value, reward4_sota[2], linewidth=3.0, label=label_sota[2], color='green')
    ax8.plot(x_value, reward4_sota[3], linewidth=3.0, label=label_sota[3], color='c')
    ax8.plot(x_value, reward4_sota[4], linewidth=3.0, label=label_sota[4], color='m')
    ax8.plot(x_value, reward4_sota[5], linewidth=3.0, label=label_sota[5], color='y')
    ax8.plot(x_value, reward4_sota[6], linewidth=3.0, label=label_sota[6], color='#1f77b4')
    ax8.plot(x_value, reward4_sota[7], linewidth=3.0, label=label_sota[7], color='#ff7f0e')
    ax8.plot(x_value, reward4_sota[8], linewidth=3.0, label=label_sota[8], color='red')
    ax8.set_ylabel(r'Average reward per step', fontsize=15)
    ax8.set_xlabel(r'Steps (x500)', fontsize=15)
    ax8.set_xlim(0, len(reward4_sota[0]))
    ax8.set_ylim(-1, -0.48)
    ax8.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax8.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.3)
    ax8.yaxis.set_major_locator(yy)
    ax8.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax8.yaxis.get_offset_text().set_fontsize(15)
    ax8.grid()
    ax8.legend(fontsize=12, loc="upper left", handlelength=1)
    ax8.set_title(label=r'(h) SOTA in Gridworld 4x4', fontsize=15, y=-0.4)


    plt.savefig("./TableQ.pdf", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()



def get_Q_value(dir):
    log = open(dir, 'r').readlines()
    max_Q_S = log[-1][:]
    max_Q_S = ast.literal_eval(max_Q_S)
    return max_Q_S


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
    Q1_learning_dir = r'../Multiarms/Result/log.Q 2025.03.14.16.35.44'
    y1_value = get_Q_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Multiarms/Result/log.AQ (1,4) 2025.03.15.16.08.02'
    y2_value = get_Q_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Multiarms/Result/log.AQ (2,4) 2025.03.15.16.11.26'
    y3_value = get_Q_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Multiarms/Result/log.AQ (4,4) 2025.03.15.15.58.28'
    y4_value = get_Q_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Multiarms/Result/log.AQ (8,4) 2025.03.15.16.14.51'
    y5_value = get_Q_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Multiarms/Result/log.AQ (16,4) 2025.03.15.16.18.11'
    y6_value = get_Q_value(
        dir=Q6_learning_dir)
    multiarm_m = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value]

    print("multiarm_m:")
    for i in multiarm_m:
        print(i[-1])



    Q1_learning_dir = r'../Multiarms/Result/log.DQ 2025.03.14.16.35.40'
    y1_value = get_Q_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Multiarms/Result/log.AQ (4,1) 2025.03.15.15.52.01'
    y2_value = get_Q_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Multiarms/Result/log.AQ (4,2) 2025.03.15.15.55.15'
    y3_value = get_Q_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Multiarms/Result/log.AQ (4,4) 2025.03.15.15.58.28'
    y4_value = get_Q_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Multiarms/Result/log.AQ (4,8) 2025.03.15.16.01.43'
    y5_value = get_Q_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Multiarms/Result/log.AQ (4,16) 2025.03.15.16.04.53'
    y6_value = get_Q_value(
        dir=Q6_learning_dir)
    multiarm_n = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value]

    print("multiarm_n:")
    for i in multiarm_n:
        print(i[-1])



    Q1_learning_dir = r'../Multiarms/Result/log.Q 2025.03.14.16.35.44'
    y1_value = get_Q_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Multiarms/Result/log.DQ 2025.03.14.16.35.40'
    y2_value = get_Q_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Multiarms/Result/log.AQ (4,8) 2025.03.15.16.01.43'
    y3_value = get_Q_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Multiarms/Result/log.SoftAQ (1) 2025.03.14.16.02.57'
    y4_value = get_Q_value(
        dir=Q4_learning_dir)
    multiarm_c = [y1_value, y2_value, y3_value, y4_value]

    print("multiarm_c:")
    for i in multiarm_c:
        print(i[-1])




    Q1_learning_dir = r'../Multiarms/Result/log.Q 2025.03.14.16.35.44'
    y1_value = get_Q_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Multiarms/Result/log.DQ 2025.03.14.16.35.40'
    y2_value = get_Q_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Multiarms/Result/log.SoftAQ (1) 2025.03.14.16.02.57'
    y3_value = get_Q_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Multiarms/Result/log.SoftAQ (2) 2025.03.14.16.09.29'
    y4_value = get_Q_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Multiarms/Result/log.SoftAQ (5) 2025.03.14.16.15.49'
    y5_value = get_Q_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Multiarms/Result/log.SoftAQ (10) 2025.03.14.16.22.09'
    y6_value = get_Q_value(
        dir=Q6_learning_dir)
    Q7_learning_dir = r'../Multiarms/Result/log.SoftAQ (100) 2025.03.14.16.28.26'
    y7_value = get_Q_value(
        dir=Q7_learning_dir)
    multiarm_a = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value, y7_value]

    print("multiarm_a:")
    for i in multiarm_a:
        print(i[-1])







    Q1_learning_dir = r'../Multiarms/Result/log.EnsembleQ 2025.03.14.18.41.09'
    y1_value = get_Q_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Multiarms/Result/log.MaxminQ 2025.03.14.18.36.36'
    y2_value = get_Q_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Multiarms/Result/log.SelfCorrectingQ 2025.03.14.18.26.50'
    y3_value = get_Q_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Multiarms/Result/log.SoftmaxQ 2025.03.14.18.23.09'
    y4_value = get_Q_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Multiarms/Result/log.WeightedDoubleQ 2025.03.14.20.16.11'
    y5_value = get_Q_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Multiarms/Result/log.REDQ 2025.03.14.19.41.06'
    y6_value = get_Q_value(
        dir=Q6_learning_dir)
    Q7_learning_dir = r'../Multiarms/Result/log.EBQL 2025.03.14.19.40.47'
    y7_value = get_Q_value(
        dir=Q7_learning_dir)
    Q8_learning_dir = r'../Multiarms/Result/log.AdaEQ 2025.03.14.19.40.21'
    y8_value = get_Q_value(
        dir=Q8_learning_dir)
    Q9_learning_dir = r'../Multiarms/Result/log.SoftAQ (1) 2025.03.14.16.02.57'
    y9_value = get_Q_value(
        dir=Q9_learning_dir)
    multiarm_sota = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value, y7_value, y8_value, y9_value]
    print("multiarm_sota:")
    for i in multiarm_sota:
        print(i[-1])

    Q1_learning_dir = r'../Roulette/Result/log.EnsembleQ 2025.03.17.14.44.12'
    y1_value = get_P_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Roulette/Result/log.MaxminQ 2025.03.21.22.39.42'
    y2_value = get_P_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Roulette/Result/log.SelfCorrectingQ 2025.03.17.14.44.57'
    y3_value = get_P_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Roulette/Result/log.SoftmaxQ 2025.03.17.14.45.50'
    y4_value = get_P_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Roulette/Result/log.WeightedDoubleQ 2025.03.17.14.46.01'
    y5_value = get_P_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Roulette/Result/log.REDQ 2025.03.17.14.44.42'
    y6_value = get_P_value(
        dir=Q6_learning_dir)
    Q7_learning_dir = r'../Roulette/Result/log.EBQL 2025.03.21.11.28.37'
    y7_value = get_P_value(
        dir=Q7_learning_dir)
    Q8_learning_dir = r'../Roulette/Result/log.AdaEQ 2025.03.17.14.42.37'
    y8_value = get_P_value(
        dir=Q8_learning_dir)
    Q9_learning_dir = r'../Roulette/Result/log.SoftAQ (1) 2025.03.16.13.40.30'
    y9_value = get_P_value(
        dir=Q9_learning_dir)
    policy_sota = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value, y7_value, y8_value, y9_value]
    print("policy_sota:")
    for i in policy_sota:
        print(i[-1])



    Q1_learning_dir = r'../Gridworld3/Result/log.EnsembleQ 2025.03.15.18.17.14'
    y1_value = get_reward_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Gridworld3/Result/log.MaxminQ 2025.03.15.22.26.24'
    y2_value = get_reward_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Gridworld3/Result/log.SelfCorrectingQ 2025.03.15.18.22.26'
    y3_value = get_reward_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Gridworld3/Result/log.SoftmaxQ 2025.03.15.18.23.57'
    y4_value = get_reward_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Gridworld3/Result/log.WeightedDoubleQ 2025.03.15.18.56.09'
    y5_value = get_reward_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Gridworld3/Result/log.REDQ 2025.03.15.20.00.21'
    y6_value = get_reward_value(
        dir=Q6_learning_dir)
    Q7_learning_dir = r'../Gridworld3/Result/log.EBQL 2025.03.15.19.56.56'
    y7_value = get_reward_value(
        dir=Q7_learning_dir)
    Q8_learning_dir = r'../Gridworld3/Result/log.AdaEQ 2025.03.15.18.52.17'
    y8_value = get_reward_value(
        dir=Q8_learning_dir)
    Q9_learning_dir = r'../Gridworld3/Result/log.SoftAQ (1) 2025.03.15.14.21.18'
    y9_value = get_reward_value(
        dir=Q9_learning_dir)
    reward3_sota = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value, y7_value, y8_value, y9_value]

    print("reward3_sota:")
    for i in reward3_sota:
        print(i[-1])

    Q1_learning_dir = r'../Gridworld4/Result/log.EnsembleQ 2025.03.16.11.44.52'
    y1_value = get_reward_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = r'../Gridworld4/Result/log.MaxminQ 2025.03.15.22.43.47'
    y2_value = get_reward_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = r'../Gridworld4/Result/log.SelfCorrectingQ 2025.03.16.11.45.06'
    y3_value = get_reward_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = r'../Gridworld4/Result/log.SoftmaxQ 2025.03.16.11.45.10'
    y4_value = get_reward_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = r'../Gridworld4/Result/log.WeightedDoubleQ 2025.03.16.11.45.13'
    y5_value = get_reward_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = r'../Gridworld4/Result/log.REDQ 2025.03.16.12.42.19'
    y6_value = get_reward_value(
        dir=Q6_learning_dir)
    Q7_learning_dir = r'../Gridworld4/Result/log.EBQL 2025.03.15.22.45.04'
    y7_value = get_reward_value(
        dir=Q7_learning_dir)
    Q8_learning_dir = r'../Gridworld4/Result/log.AdaEQ 2025.03.15.22.42.48'
    y8_value = get_reward_value(
        dir=Q8_learning_dir)
    Q9_learning_dir = r'../Gridworld4/Result/log.SoftAQ (1) 2025.03.15.22.43.09'
    y9_value = get_reward_value(
        dir=Q9_learning_dir)
    reward4_sota = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value, y7_value, y8_value, y9_value]

    print("reward4_sota:")
    for i in reward4_sota:
        print(i[-1])


    figure(multiarm_m, multiarm_n, multiarm_c, multiarm_a, multiarm_sota, policy_sota, reward3_sota, reward4_sota)

