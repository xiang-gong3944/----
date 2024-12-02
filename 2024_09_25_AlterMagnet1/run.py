import numpy as np
import matplotlib.pyplot as plt
from lib import funcs as fs
from lib import post

def make_chi_graph(x, y, Ne, label):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlim(0.55,1.45)
    ax1.set_ylime(0.4, -0.8)
    ax1.hlines(0, 0.55, 1.45, color="grey")
    ax1.set_title("Ne = {:.1f}".format(Ne))
    ax1.set_xlabel("U (eV)")
    ax1.set_ylabel(label)
    ax1.plot(x, y)

    # 出力
    image_path = "./output/{}/Ne{:d}.png".format(label, int(Ne*10))
    plt.savefig(image_path)

    return image_path

# 本文
if __name__ == '__main__':
    model = fs.KappaET2X(0.01, 7.5, 11)
    model.calc_scf()
    model.calc_nscf()
    # model.plot_dos()
    # model.calc_spin_conductivity("x", "y")
    model.plot_fermi_surface()
    # model.plot_band()