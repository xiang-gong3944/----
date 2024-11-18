import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import plotly.graph_objects as go
import scipy.linalg
from matplotlib.colors import LinearSegmentedColormap


k_points = {}
k_points["Γ"]       = [0.0, 0.0]
k_points["X"]        = [np.pi, 0.0]
k_points["Y"]        = [0.0, np.pi]
k_points["M"]        = [np.pi, np.pi]
k_points["Σ"]       = [np.pi/2, np.pi/2]
k_points["M'"]       = [-np.pi, np.pi]
k_points["Σ'"]      = [-np.pi/2, np.pi/2]

# ホッピングパラメータ
# 各ホッピングの果たしている役割を見るために t = 0 としてみたい
ta = -0.207     # eV
tb = -0.067     # eV
tp = -0.102     # eV
tq = 0.043      # eV

# バンドをプロットする際の経路を生成する
def gen_kpath(path, npoints = 50):
    """バンド図を書くときの対称点に沿った波数ベクトルの列を作る

    Args:
        path (list[tuple[str, str]]): プロットする対称点と終点の tuple 列
        npoints (int, optional): 対称点の間の点の数 Defaults to 50.

    Returns:
        _type_: 経路の座標、プロット用のラベル、プロット用のラベルの位置、プロット用のx軸の値
    """
    k_path = []
    labels = []
    labels_loc = []
    distances = []
    total_distance = 0.0
    for (spoint, epoint) in path :
        k_start = k_points[spoint]
        k_end   = k_points[epoint]
        # 線形補完でnpoints個のk点の生成
        segment = np.linspace(k_start, k_end, npoints)
        k_path.extend(segment)

        labels.append(spoint)
        labels_loc.append(total_distance)

        distance = np.linalg.norm(np.array(k_end)-np.array(k_start))
        segment_dist = np.linspace(total_distance, total_distance+distance, npoints)
        distances.extend(segment_dist)
        total_distance += distance

    labels.append(path[-1][1])
    labels_loc.append(total_distance)

    return k_path, labels, labels_loc, distances

def Hamiltonian(kx, ky, U=0.0, Delta=0.8):
    """ある波数のでのハミルトニアン
    Args:
        (float) kx: 波数のx成分
        (float) ky: 波数のy成分
        (float) U: オンサイト相互作用の強さ
        (float) delta: 反強磁性分子場の強さ

    Returns:
        ハミルトニアンの固有値[0]と固有ベクトルの行列[1]
    """

    # ホッピング項
    H = np.zeros((8,8), dtype=np.complex128)
    H[0,1] = ta + tb*np.exp(-1j*kx)                          # A1up   from A2up
    H[0,2] = tq * (1 + np.exp(1j*ky))                        # A1up   from B1up
    H[0,3] = tp * np.exp(1j*ky) * (1 + np.exp(1j*kx))        # A1up   from B2up

    H[1,2] = tp * (1 + np.exp(1j*kx))                      # A2up   from B1up
    H[1,3] = tq * np.exp(1j*kx) * (1 + np.exp(1j*ky))      # A2up   from B2up

    H[2,3] = ta + tb*np.exp(1j*kx)                         # B1up   from B2up

    H[4,5] = H[0,1]                                         # A1down from A2down
    H[4,6] = H[0,2]                                         # A1down from B1down
    H[4,7] = H[0,3]                                         # A1down from B2down

    H[5,6] = H[1,2]                                         # A2down from B1down
    H[5,7] = H[1,3]                                         # A2down from B2down

    H[6,7] = H[2,3]                                         # B1down from B2down

    #エルミート化
    for i in range(1,8):
        for j in range(0, i):
            H[i][j] = H[j][i].conjugate()

    # 反強磁性分子内磁場を表すハートリー項
    H[0,0] = - U * Delta / 4    # A1 up
    H[1,1] = - U * Delta / 4    # A2 up
    H[2,2] = + U * Delta / 4    # B1 up
    H[3,3] = + U * Delta / 4    # B2 up
    H[4,4] = + U * Delta / 4    # A1 down
    H[5,5] = + U * Delta / 4    # A2 down
    H[6,6] = - U * Delta / 4    # B1 down
    H[7,7] = - U * Delta / 4    # B2 down

    return scipy.linalg.eigh(H)

def Current(kx, ky, mu):
    """ある波数での電流演算子行列

    Args:
        kx (float): 波数のx成分
        ky (float): 波数のy成分
        mu (sring): 電流の方向. "x", "y", "z" のみ受け付ける

    Return:
        J (ndarray): 8x8の電流演算子行列
    """

    if (mu == "x"):
        J = np.zeros((8,8), dtype=np.complex128)

        J[0,1] =-1j * tb * np.exp(-1j*kx)                           # A1up   from A2up
        J[0,3] = 1j * tp * np.exp(1j*(kx+ky))                       # A1up   from B2up

        J[1,2] = 1j * tp * np.exp(1j*kx)                            # A2up   from B1up
        J[1,3] = 1j * tq * np.exp(1j*kx) * (1 + np.exp(1j*ky))      # A2up   from B2up

        J[2,3] = 1j * tb*np.exp(1j*kx)                              # B1up   from B2up

        J[4,5] = J[0,1]                                         # A1down from A2down
        J[4,7] = J[0,3]                                         # A1down from B2down

        J[5,6] = J[1,2]                                         # A2down from B1down
        J[5,7] = J[1,3]                                         # A2down from B2down

        J[6,7] = J[2,3]                                         # B1down from B2down

    elif (mu == "y"):
        J = np.zeros((8,8), dtype=np.complex128)

        J[0,2] = 1j * tq * np.exp(1j*ky)                             # A1up   from B1up
        J[0,3] = 1j * tp * np.exp(1j*ky) * (1 + np.exp(1j*kx))      # A1up   from B2up

        J[1,3] = 1j * tq * np.exp(1j*(kx + ky))                      # A2up   from B2up

        J[4,6] = J[0,2]                                         # A1down from B1down
        J[4,7] = J[0,3]                                         # A1down from B2down

        J[5,7] = J[1,3]                                         # A2down from B2down


    elif (mu == "z"):
        J = np.zeros((8,8), dtype=np.complex128)

    else :
        print("The current direction is incorrect.")
        return

    #エルミート化
    for i in range(1,8):
        for j in range(0, i):
            J[i][j] = J[j][i].conjugate()

    return -J

def SpinCurrent(kx, ky, mu):
    """ある波数での電流演算子行列

    Args:
        kx (float): 波数のx成分
        ky (float): 波数のy成分
        mu (sring): 電流の方向. "x", "y", "z" のみ受け付ける

    Return:
        J (ndarray): 8x8の電流演算子行列
    """

    if (mu == "x"):
        J = np.zeros((8,8), dtype=np.complex128)

        J[0,1] =-1j * tb * np.exp(-1j*kx)                           # A1up   from A2up
        J[0,3] = 1j * tp * np.exp(1j*(kx+ky))                       # A1up   from B2up

        J[1,2] = 1j * tp * np.exp(1j*kx)                            # A2up   from B1up
        J[1,3] = 1j * tq * np.exp(1j*kx) * (1 + np.exp(1j*ky))      # A2up   from B2up

        J[2,3] = 1j * tb*np.exp(1j*kx)                              # B1up   from B2up

        J[4,5] =-J[0,1]                                         # A1down from A2down
        J[4,7] =-J[0,3]                                         # A1down from B2down

        J[5,6] =-J[1,2]                                         # A2down from B1down
        J[5,7] =-J[1,3]                                         # A2down from B2down

        J[6,7] =-J[2,3]                                         # B1down from B2down

    elif (mu == "y"):
        J = np.zeros((8,8), dtype=np.complex128)

        J[0,2] = 1j * tq * np.exp(1j*ky)                             # A1up   from B1up
        J[0,3] = 1j * tp * np.exp(1j*ky) * (1 + np.exp(1j*kx))      # A1up   from B2up

        J[1,3] = 1j * tq * np.exp(1j*(kx + ky))                      # A2up   from B2up

        J[4,6] =-J[0,2]                                         # A1down from B1down
        J[4,7] =-J[0,3]                                         # A1down from B2down

        J[5,7] =-J[1,3]                                         # A2down from B2down


    elif (mu == "z"):
        J = np.zeros((8,8), dtype=np.complex128)

    else :
        print("The current direction is incorrect.")
        return

    #エルミート化
    for i in range(1,8):
        for j in range(0, i):
            J[i][j] = J[j][i].conjugate()

    return J/2

def calc_delta(N_site):
    """反強磁性磁化の大きさ

    Args:
        N_site (list[float]): 各サイトにある電子数

    Returns:
       delta (float): 反強磁性磁化の大きさ
    """
    delta = np.abs((N_site[0] + N_site[1]) - (N_site[2] + N_site[3]) - (N_site[4] + N_site[5]) + (N_site[6] + N_site[7])) / 2
    #               A1_up       A2_up         B1_up       B2_up         A1_down     A2_down       B1_down     B2_down
    return delta

def Steffensen(array):
    """Steffensen の反復法で収束を早めるための処理

    Args:
        array : SCF 計算で出てきた2つの値

    Returns:
        val : 収束が早い数列の値
    """
    # 収束すると分母が0になり発散するため例外処理が必要
    try:
        res = array[-3] - (array[-3]-array[-2])**2 / (array[-3] - 2*array[-2] + array[-1])
        return res
    except:
        return array[-2]

def calc_spin(enes, eigenstate):
    """各サイトのスピンの大きさ

    Args:
        enes: ある波数の固有エネルギー
        eigenstate :  ある波数の固有ベクトル

    Returns:
        spin : 各サイトのスピンの大きさ
    """
    spin = []
    for l in range(8):
        sp = (np.abs(eigenstate[0,l])**2    # A1_up
                +np.abs(eigenstate[1,l])**2    # A2_up
                +np.abs(eigenstate[2,l])**2    # B1_up
                +np.abs(eigenstate[3,l])**2    # B2_up
                -np.abs(eigenstate[4,l])**2    # A1_down
                -np.abs(eigenstate[5,l])**2    # A2_down
                -np.abs(eigenstate[6,l])**2    # B1_down
                -np.abs(eigenstate[7,l])**2    # B2_donw
                )
        spin.append(sp)
    for l in range(4):
        if(np.abs(enes[2*l]-enes[2*l + 1])<0.000001):
            spin[2*l] = 0
            spin[2*l+1] = 0

    return np.array(spin)

class KappaET2X:
    def __init__(self, U, Ne=6.0, k_mesh=31):
        """モデルのパラメータの設定

        Args:
            U (float): オンサイト相互作用の大きさ
            Ne (float, optional): 単位胞内での電子の数 Defaults to 6.0.
            k_mesh (int, optional): k点の細かさ Defaults to 31.
        """
        self.U          = U
        self.Ne         = Ne
        self.k_mesh     = k_mesh

        ne1 = Ne/8.0 + 0.2
        ne2 = Ne/8.0 - 0.2
        self.N_site_scf = np.array([
                        [ne1, ne1, ne2, ne2, ne2, ne2, ne1, ne1]])
        self.Ef_scf     = np.array([])
        self.Delta_scf  = np.array([0.8])
        self.Etot_scf   = np.array([0.8])
        self.Ntot_scf   = np.array([Ne])

        self.delta      = 0
        self.ef         = 0
        self.enes       = np.zeros((k_mesh, k_mesh, 8))
        self.eigenStates= np.zeros((k_mesh, k_mesh, 8, 8), dtype=np.complex128)
        self.spins      = np.zeros((k_mesh, k_mesh, 8))

        self.path       = [("Γ", "Y"), ("Y", "M'"), ("M'", "Σ'"), ("Σ'","Γ"),
                           ("Γ", "Σ"), ("Σ", "M"), ("M", "X"), ("X", "Γ")]

        self.E          = 0
        self.dos        = np.array([])



    def calc_scf(self, iteration = 100, err = 1e-6):
        """自己無頓着計算を行う。delta と ef を決定する。

        Args:
            iteration (int, optional): 繰り返す回数の上限. Defaults to 100.
            err (float, optional): 収束条件. Defaults to 1e-6.
        """

        # 一度やったらもうやらない。
        if(self.Ef_scf.size > 1):
            print("SCF calculation was already done.")
            return

        print("SCF calculation start.")

        kx = np.linspace(-np.pi, np.pi, self.k_mesh)
        ky = np.linspace(-np.pi, np.pi, self.k_mesh)
        kx, ky = np.meshgrid(kx, ky)

        # ここから自己無頓着方程式のループになる
        for scf_iteration in range(iteration):
            # Steffensen の反復法
            for m in range(2):
                # フェルミエネルギーを求める
                enes = []
                eigenEnes = np.zeros((self.k_mesh, self.k_mesh, 8))
                eigenStates = np.zeros((self.k_mesh, self.k_mesh, 8, 8), dtype=np.complex128)
                Delta   = calc_delta(self.N_site_scf[-1])

                # ブリュアンゾーン内の全探査
                for i in range(self.k_mesh):
                    for j in range(self.k_mesh):
                        eigenEnergy, eigenState = Hamiltonian(kx[i][j],ky[i][j], self.U, Delta)
                        enes = np.append(enes, eigenEnergy)
                        eigenEnes[i,j] = eigenEnergy
                        eigenStates[i,j] = eigenState

                # 求めたエネルギー固有値をソートして下から何番目というのを探してやればよい
                # 絶縁体相のときのことを考えると平均をとる必要がある
                sorted_enes = np.sort(enes)
                ef = (sorted_enes[int(self.k_mesh * self.k_mesh * self.Ne) - 1]
                      + sorted_enes[int(self.k_mesh * self.k_mesh * self.Ne)])/2
                self.Ef_scf = np.append(self.Ef_scf, ef)

                # scf で求める値の初期化
                nsite  = np.zeros((8))
                etot   = 0
                ntot   = 0

                # ブリュアンゾーン内の全探査
                for i in range(self.k_mesh):
                    for j in range(self.k_mesh):
                        # フェルミ分布で重みづけして足す。今は絶対零度なので階段関数になってる。
                        for l in range(8):
                            if (eigenEnes[i,j,l] <= ef) :
                                nsite += np.abs(eigenStates[i,j][:,l])**2
                                etot  += eigenEnes[i,j,l]
                                # ntot  += 1.0          # ntot を計算するとなぜか 0 除算が発生する。

                # 規格化して足す
                nsite /= self.k_mesh * self.k_mesh
                self.N_site_scf = np.vstack((self.N_site_scf, nsite))

                self.Delta_scf = np.append(self.Delta_scf,calc_delta(nsite))

                etot /= self.k_mesh * self.k_mesh
                self.Etot_scf = np.append(self.Etot_scf, etot)

                # ntot /= self.k_mesh * self.k_mesh
                # self.Ntot_scf = np.append(self.Ntot_scf, ntot)

            # Steffensen の反復法
            ef = Steffensen(self.Ef_scf)
            self.Ef_scf = np.append(self.Ef_scf, ef)

            nsite = Steffensen(self.N_site_scf)
            self.N_site_scf = np.vstack((self.N_site_scf, nsite))

            delta = Steffensen(self.Delta_scf)
            self.Delta_scf = np.append(self.Delta_scf, delta)

            etot = Steffensen(self.Etot_scf)
            self.Etot_scf = np.append(self.Etot_scf, etot)

            # ntot = Steffensen(self.Ntot_scf)
            # self.Ntot_scf = np.append(self.Ntot_scf, ntot)

            # 与えられた誤差いないに収まったら終了する
            if(np.abs(self.Delta_scf[-1]-self.Delta_scf[-4]) < err) :

                self.delta = self.Delta_scf[-1]
                self.ef    = self.Ef_scf[-1]

                print("SCF loop converged. U = {:.2f}, Ne = {:1.2f}, err < {:1.1e}, loop = {:2d}, delta = {:1.2e}".format(self.U, self.Ne, err, scf_iteration*3, self.delta))
                print("")

                return

        # 収束しなかったときの処理
        self.delta = self.Delta_scf[-1]
        self.ef    = self.Ef_scf[-1]
        print('\033[41m'+"Calculation didn't converge. err > {:1.1e}, U = {:.2f}, Ne = {:1.2f} loop = {:2d}, delta = {:1.2e}".format(err, self.U, self. Ne, iteration*3, self.delta)+'\033[0m')
        print(f"latter deltas are {self.Delta_scf[-4:-1]}")
        print("")


        return


    def calc_nscf(self):
        """
        delta と ef が与えられたときの各k点の固有状態のエネルギー、状態ベクトル、スピンの大きさの計算をする
        """

        print("NSCF calculation start.")

        # ブリュアンゾーンのメッシュの生成
        kx = np.linspace(-np.pi, np.pi, self.k_mesh)
        ky = np.linspace(-np.pi, np.pi, self.k_mesh)
        kx, ky = np.meshgrid(kx, ky)

        # フェルミ準位を求めるためのリスト
        sorted_enes = np.array([])

        # メッシュの各点でのエネルギー固有値の計算
        for i in range(self.k_mesh):
            for j in range(self.k_mesh):
                enes, eigenstate = Hamiltonian(kx[i][j],ky[i][j], self.U, self.delta)
                spin = calc_spin(enes, eigenstate)
                self.enes[i,j]         = enes
                self.eigenStates[i,j]  = eigenstate
                self.spins[i,j]        = np.array(spin)

                sorted_enes = np.append(sorted_enes, enes)

        sorted_enes = np.sort(sorted_enes)
        self.ef = (sorted_enes[int(self.k_mesh * self.k_mesh * self.Ne) - 1] + sorted_enes[int(self.k_mesh * self.k_mesh * self.Ne)])/2

        print("NSCF calculation finished.")
        print("")



    def calc_dos(self, E_fineness=1000, sigma2 = 0.0001):

        E = np.linspace(np.min(self.enes)-0.1, np.max(self.enes)+0.1, E_fineness)
        for e in E:
            self.dos = np.append(self.dos, np.sum(np.exp(-(e-self.enes)**2 / 2 / sigma2 ) / np.sqrt(2 * np.pi * sigma2)))
        self.dos /= np.sum(self.dos)



    def calc_spinConductivity(self, mu, nu, gamma=0.0001):
        if(self.Ef_scf.size < 2):
            print("SCF calculation wasn't done yet.")
            return

        print("SpinConductivity calculation start.")
        # スピン伝導度 複素数として初期化
        chi = 0.0 + 0.0*1j

        # ブリュアンゾーンのメッシュの生成
        kx = np.linspace(-np.pi, np.pi, self.k_mesh)
        ky = np.linspace(-np.pi, np.pi, self.k_mesh)
        kx, ky = np.meshgrid(kx, ky)

        # ブリュアンゾーンの和
        for i in range(self.k_mesh):
            for j in range(self.k_mesh):
                # enes, eigenstate = Hamiltonian(kx[i][j],ky[i][j], self.U, self.delta)
                # 各波数におけるそれぞれの固有状態の和
                Js_matrix = np.conjugate(self.eigenStates[i,j][:].T) @ SpinCurrent(kx[i,j], ky[i,j], mu) @ self.eigenStates[i,j][:]
                J_matrix  = np.conjugate(self.eigenStates[i,j][:].T) @     Current(kx[i,j], ky[i,j], mu) @ self.eigenStates[i,j][:]
                for m in range(8):
                    for n in range(8):
                        #零除算を避ける
                        if(np.abs(self.enes[i,j][m]-self.enes[i,j][n])<0.000001):
                            continue

                        # フェルミ分布
                        efm = 1 if (self.enes[i,j][m]<self.ef) else 0
                        efn = 1 if (self.enes[i,j][n]<self.ef) else 0

                        # Js = (self.eigenStates[i,j][:,m].conj()) @ SpinCurrent(kx[i,j], ky[i,j], mu) @ self.eigenStates[i,j][:,n]
                        # J  = (self.eigenStates[i,j][:,n].conj()) @     Current(kx[i,j], ky[i,j], nu) @ self.eigenStates[i,j][:,m]

                        Js = Js_matrix[m,n]
                        J  =  J_matrix[n,m]

                        # Js = (eigenstate[:,m].conj()) @ SpinCurrent(kx[i,j], ky[i,j], mu) @ eigenstate[:,n]
                        # J  = (eigenstate[:,n].conj()) @     Current(kx[i,j], ky[i,j], nu) @ eigenstate[:,m]

                        chi += Js * J * (efm - efn) / (self.enes[i,j][m]-self.enes[i,j][n]+1j*gamma)**2

        chi /= (self.k_mesh*self.k_mesh*1j)

        print("Spin Conductivity calculation finished")
        print("ReChi = {:1.2e}, ImChi = {:1.2e}".format(np.real(chi), np.imag(chi)))
        print("")

        return chi



    def plot_Nsite(self):
        if(self.Ef_scf.size < 2):
            print("SCF calculation wasn't done yet.")
            return

        plt.figure(figsize=[12.8,4.8])
        plt.subplot(121)
        for i in [0, 1, 6, 7]:
            plt.plot(self.N_site_scf[:,i], label = "site {:d} = {:.3f}".format(i, self.N_site_scf[-1, i]))
        plt.legend()
        plt.subplot(122)
        for i in [2, 3, 4, 5]:
            plt.plot(self.N_site_scf[:,i], label = "site {:d} = {:.3f}".format(i, self.N_site_scf[-1, i]))
        plt.legend()
        plt.show()



    def plot_scf(self):
        if(self.Ef_scf.size < 2):
            print("SCF calculation wasn't done yet.")
            return

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel("scf loop")

        ax1.set_ylabel("Delta")
        ax1.plot(self.Delta_scf, label="Delta = {:.5f}".format(self.Delta_scf[-1]), color = "tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Ef (eV)")
        ax2.plot(self.Ef_scf, label="Ef = {:.5f}".format(self.Ef_scf[-1]), color = "tab:orange")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2)

        plt.show()



    def plot_band(self):
        if(self.Ef_scf.size < 2):
            print("SCF calculation wasn't done yet.")
            return

        k_path, label, label_loc, distances = gen_kpath(self.path)

        bands = []
        spins = []
        # plt.xticks(label)
        for kxy in k_path:
            enes, eigenstate = Hamiltonian(kxy[0], kxy[1], self.U, self.delta)
            bands.append(enes)
            spin = calc_spin(enes, eigenstate)
            spins.append(spin)
        bands = np.array(bands)
        spins = np.array(spins)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        # ax.set_aspect(5)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] ='Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'

        plt.xlabel("k points")
        plt.ylabel("Energy (eV)")


        Ymin = np.min(bands)-0.05
        Ymax = np.max(bands)+0.05
        plt.xticks(label_loc, label)
        plt.xlim(label_loc[0], label_loc[-1])
        plt.ylim(Ymin, Ymax)

        colors = ["tab:blue", "tab:green","tab:orange"]
        cmap_name = LinearSegmentedColormap.from_list("custom",colors, 10)
        for i in range(8):
            plt.scatter(distances, bands[:,i], c=spins[:,i], cmap=cmap_name, vmin=-1, vmax=1, s=1)
        plt.vlines(label_loc[1:-1], Ymin,Ymax, "grey", "dashed")
        plt.hlines(self.ef, distances[0], distances[-1], "grey")
        plt.title("$E_f$ = {:.5f}".format(self.ef))
        plt.colorbar()

        plt.show()



    def plot3d_band(self):
    # 参考 https://qiita.com/okumakito/items/3b2ccc9966c43a5e84d0

        if(self.Ef_scf.size < 2):
            print("SCF calculation wasn't done yet.")
            return


        kx = np.linspace(-np.pi, np.pi, self.k_mesh)
        ky = np.linspace(-np.pi, np.pi, self.k_mesh)
        kx, ky = np.meshgrid(kx, ky)


        fig = go.Figure()

        contours = dict(
            x=dict(highlight=False, show=True, color='grey', start=-3.5, end=3.5, size=0.5),
            y=dict(highlight=False, show=True, color='grey', start=-3.5, end=3.5, size=0.5),
            z=dict(highlight=False, show=False, start=-1, end = 1, size=0.5)
        )

        fig.add_trace(go.Surface(
                z=self.enes[:,:,0]-self.ef,
                x=kx,
                y=ky,
                surfacecolor=self.spins[:,:,0],
                colorscale = "balance",
                cmin=-1.5,
                cmax=1.5,
                showscale = False,
                hoverinfo="skip",
                # opacity=0.8,
                # hidesurface=True,
            )
        )
        for i in range(1, 8):
            fig.add_trace(go.Surface(
                    z=self.enes[:,:,i]-self.ef,
                    x=kx,
                    y=ky,
                    surfacecolor=self.spins[:,:,i],
                    colorscale = "balance",
                    cmin=-1.5,
                    cmax=1.5,
                    showscale = False,
                    hoverinfo="skip",
                    contours=contours,
                    # opacity=0.8,
                    # hidesurface=True,
                )
            )

        axis = dict(visible=True)
        fig.update_scenes(
            xaxis=axis,
            yaxis=axis,
            zaxis=axis,
            aspectratio=dict(x=1,y=1,z=1.5)
        )
        fig.update_layout(
            width=800,   # グラフの幅
            height=800   # グラフの高さ
        )
        fig.show()

    def plot_dos(self):
        E = np.linspace(np.min(self.enes)-0.1, np.max(self.enes)+0.1, self.dos.size)

        ysacale = np.max(self.dos)
        plt.ylim(-0.04*ysacale, 1.04*ysacale)

        plt.plot(E, self.dos)
        plt.vlines(self.ef, -0.04*ysacale, 1.04*ysacale, color="gray", linestyles="dashed")
        plt.title("Ef={:.2f} eV".format(self.ef))
        plt.show()
