import os
import matplotlib.pyplot as plt
import numpy as np

#cd の移動
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# matplotlibの初期設定
plt_config = {
    "figure.dpi": 150,
    "font.size": 14,
    "figure.subplot.bottom": 0.15,
    "figure.subplot.left": 0.17,
    "axes.xmargin": 0,
    "axes.ymargin": 0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.cal": "serif",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif.bold",
    "mathtext.fontset": "cm",
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",
    "legend.fancybox": False
}
plt.rcParams.update(plt_config)


# データ
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# プロット
plt.plot(x, y, ".", ls="", label="$\sin{(t)}$")

# 軸ラベル
plt.xlabel("Time[s]")
plt.ylabel("Amplitude[V]")

# x, y軸の範囲を指定
# plt.xlim(0, 2 * np.pi)
# plt.ylim(-1, 1)

plt.legend()
plt.show()

# #出力
plt.savefig("../image/graph/fig1.png")