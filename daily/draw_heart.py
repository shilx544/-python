import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from numpy import *
def draw_heart():
    x1 = linspace(-5.1, 0, 10000)
    x2 = linspace(0, 5.1, 10000)
    x3 = linspace(-5, 5, 10000)
    # y1 = sqrt(2*sqrt(power(x1, 2))-power(x1,2))
    # y2 = power(sin(abs(x2)-1), -1) - math.pi/2

    #函数表达式
    y1 = sqrt(6.3-power((x1+2.5), 2))
    y2 = sqrt(6.3-power((x2-2.5), 2))
    y3 = (-2) * sqrt(5-abs(x3))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    #开启副刻度
    plt.minorticks_on()
    #设置主刻度
    xmajorLocator = tck.MultipleLocator(1.0)
    ymajorLocator = tck.MultipleLocator(1.0)

    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    #设置坐标轴范围
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    #设置坐标位置
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    # xmajorFormatter = tck.FormatStrFormatter('%1.1f')
    # ymajorFormatter = tck.FormatStrFormatter('%3.1f')
    # ax.xaxis.set_major_formatter(xmajorFormatter)
    # ax.yaxis.set_major_formatter(ymajorFormatter)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.plot(x1, y1, color="black", linewidth=2)
    plt.plot(x2, y2, color="black", linewidth=2)
    plt.plot(x3, y3, color="black", linewidth=2)
    plt.annotate(' y1 = sqrt(6.3-power((x1+2.5), 2))', xy=(-3,3),xytext=(-6.5,3))
    plt.grid()
    plt.savefig('heart.png')
    plt.show()

if __name__ == '__main__':
    draw_heart()