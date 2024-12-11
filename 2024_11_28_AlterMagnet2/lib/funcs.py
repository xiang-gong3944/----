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

# ひずみ具合
a = 1.0
b = 1.0
# Slater-Koster パラメータ
Vpps = 1.0
Vppp = -0.6
Vpds = 1.85

# ホッピングパラメータ
Tpp = a * b * (Vpps-Vppp) / (a**2 + b**2)
aaaa = 0.5
Tpd1 = np.sqrt(3) * Vpds / 2 * (1 + aaaa)    # 短いホッピング
Tpd2 = np.sqrt(3) * Vpds / 2 * (1 - aaaa)   # 長いホッピング

# クーロン相互作用の強さ
Ud = 8.0
Up = 4.0
Upd = 1.0

# 軌道エネルギー差
Ep = 3.0

# 軌道の数
n_orbit = 6

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
    # del spoint, epoint

    labels.append(path[-1][1])
    labels_loc.append(total_distance)

    return k_path, labels, labels_loc, distances

def Hamiltonian(kx, ky, delta=0):
    """ある波数のでのハミルトニアン
    Args:
        (float) kx: 波数のx成分
        (float) ky: 波数のy成分
        (float) U: オンサイト相互作用の強さ
        (float) delta: 反強磁性分子場の強さ

    Returns:
        ハミルトニアンの固有値[0]と固有ベクトルの行列[1]
    """

    H = np.zeros((n_orbit*2, n_orbit*2), dtype=np.complex128)

    # ホッピング項
    H[0,2] = Tpd1 * np.exp(1j*ky/4)
    H[0,3] = Tpd2 * np.exp(1j*kx/4)
    H[0,4] = Tpd1 * np.exp(-1j*ky/4)
    H[0,5] = Tpd2 * np.exp(-1j*kx/4)

    H[1,2] = Tpd2 * np.exp(-1j*ky/4)
    H[1,3] = Tpd1 * np.exp(-1j*kx/4)
    H[1,4] = Tpd2 * np.exp(1j*ky/4)
    H[1,5] = Tpd1 * np.exp(1j*kx/4)

    H[2,3] = -2*Tpp * np.cos((kx-ky)/4)
    H[2,5] = -2*Tpp * np.cos((kx+ky)/4)

    H[3,4] = -2*Tpp * np.cos((kx+ky)/4)

    H[4,5] = -2*Tpp * np.cos((kx-ky)/4)

    #エルミート化
    for i in range(1,n_orbit*2):
        for j in range(0, i):
            H[i][j] = H[j][i].conjugate()
    del i, j

    # 軌道準位差
    H[2,2] = Ep
    H[3,3] = Ep
    H[4,4] = Ep
    H[5,5] = Ep

    # 反対向きスピンの分
    for i in range(n_orbit):
        for j in range(n_orbit):
            H[i+n_orbit,j+n_orbit] = H[i,j]
    del i, j

    # Cu イオンでのクーロン相互作用(ハートリー項のみ)
    H[0,0] = -Ud * delta /2
    H[1,1] = Ud * delta /2
    H[0+n_orbit,0+n_orbit] = Ud * delta /2
    H[1+n_orbit,1+n_orbit] = -Ud * delta /2

    return scipy.linalg.eigh(H)

# def Current(kx, ky, mu):
#     """ある波数での電流演算子行列

#     Args:
#         kx (float): 波数のx成分
#         ky (float): 波数のy成分
#         mu (sring): 電流の方向. "x", "y", "z" のみ受け付ける

#     Return:
#         J (ndarray): 8x8の電流演算子行列
#     """

#     if (mu == "x"):
#         J = np.zeros((8,8), dtype=np.complex128)

#         J[0,1] =-1j * tb * np.exp(-1j*kx)                           # A1up   from A2up
#         J[0,3] = 1j * tp * np.exp(1j*(kx+ky))                       # A1up   from B2up

#         J[1,2] = 1j * tp * np.exp(1j*kx)                            # A2up   from B1up
#         J[1,3] = 1j * tq * np.exp(1j*kx) * (1 + np.exp(1j*ky))      # A2up   from B2up

#         J[2,3] = 1j * tb*np.exp(1j*kx)                              # B1up   from B2up

#         J[4,5] = J[0,1]                                         # A1down from A2down
#         J[4,7] = J[0,3]                                         # A1down from B2down

#         J[5,6] = J[1,2]                                         # A2down from B1down
#         J[5,7] = J[1,3]                                         # A2down from B2down

#         J[6,7] = J[2,3]                                         # B1down from B2down

#     elif (mu == "y"):
#         J = np.zeros((8,8), dtype=np.complex128)

#         J[0,2] = 1j * tq * np.exp(1j*ky)                             # A1up   from B1up
#         J[0,3] = 1j * tp * np.exp(1j*ky) * (1 + np.exp(1j*kx))      # A1up   from B2up

#         J[1,3] = 1j * tq * np.exp(1j*(kx + ky))                      # A2up   from B2up

#         J[4,6] = J[0,2]                                         # A1down from B1down
#         J[4,7] = J[0,3]                                         # A1down from B2down

#         J[5,7] = J[1,3]                                         # A2down from B2down


#     elif (mu == "z"):
#         J = np.zeros((8,8), dtype=np.complex128)

#     else :
#         print("The current direction is incorrect.")
#         return

#     #エルミート化
#     for i in range(1,8):
#         for j in range(0, i):
#             J[i][j] = J[j][i].conjugate()
#     del i, j

#     return -J

# def SpinCurrent(kx, ky, mu):
#     """ある波数での電流演算子行列

#     Args:
#         kx (float): 波数のx成分
#         ky (float): 波数のy成分
#         mu (sring): 電流の方向. "x", "y", "z" のみ受け付ける

#     Return:
#         J (ndarray): 8x8の電流演算子行列
#     """

#     if (mu == "x"):
#         J = np.zeros((8,8), dtype=np.complex128)

#         J[0,1] =-1j * tb * np.exp(-1j*kx)                           # A1up   from A2up
#         J[0,3] = 1j * tp * np.exp(1j*(kx+ky))                       # A1up   from B2up

#         J[1,2] = 1j * tp * np.exp(1j*kx)                            # A2up   from B1up
#         J[1,3] = 1j * tq * np.exp(1j*kx) * (1 + np.exp(1j*ky))      # A2up   from B2up

#         J[2,3] = 1j * tb*np.exp(1j*kx)                              # B1up   from B2up

#         J[4,5] =-J[0,1]                                         # A1down from A2down
#         J[4,7] =-J[0,3]                                         # A1down from B2down

#         J[5,6] =-J[1,2]                                         # A2down from B1down
#         J[5,7] =-J[1,3]                                         # A2down from B2down

#         J[6,7] =-J[2,3]                                         # B1down from B2down

#     elif (mu == "y"):
#         J = np.zeros((8,8), dtype=np.complex128)

#         J[0,2] = 1j * tq * np.exp(1j*ky)                             # A1up   from B1up
#         J[0,3] = 1j * tp * np.exp(1j*ky) * (1 + np.exp(1j*kx))      # A1up   from B2up

#         J[1,3] = 1j * tq * np.exp(1j*(kx + ky))                      # A2up   from B2up

#         J[4,6] =-J[0,2]                                         # A1down from B1down
#         J[4,7] =-J[0,3]                                         # A1down from B2down

#         J[5,7] =-J[1,3]                                         # A2down from B2down


#     elif (mu == "z"):
#         J = np.zeros((8,8), dtype=np.complex128)

#     else :
#         print("The current direction is incorrect.")
#         return

#     #エルミート化
#     for i in range(1,8):
#         for j in range(0, i):
#             J[i][j] = J[j][i].conjugate()
#     del i, j

#     return J/2

def calc_delta(N_site):
    """反強磁性磁化の大きさ

    Args:
        N_site (list[float]): 各サイトにある電子数

    Returns:
       delta (float): 反強磁性磁化の大きさ
    """
    delta = np.abs((N_site[0] + N_site[1]) - (N_site[6] + N_site[7])) / 2
    # delta = np.abs(np.sum(N_site[:n_orbit])-np.sum(N_site[n_orbit:])) / 2

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
    for l in range(n_orbit*2):
        sp = (np.abs(eigenstate[0,l])**2
                +np.abs(eigenstate[1,l])**2
                +np.abs(eigenstate[2,l])**2
                +np.abs(eigenstate[3,l])**2
                +np.abs(eigenstate[4,l])**2
                +np.abs(eigenstate[5,l])**2
                -np.abs(eigenstate[6,l])**2
                -np.abs(eigenstate[7,l])**2
                -np.abs(eigenstate[8,l])**2
                -np.abs(eigenstate[9,l])**2
                -np.abs(eigenstate[10,l])**2
                -np.abs(eigenstate[11,l])**2
                )
        # prob = np.abs(eigenstate[:,l])**2
        # sp = prob[0] + prob[1] + prob[2] + prob[3] + prob[4] + prob[5] -(
        #     prob[6] + prob[7] + prob[8] + prob[9] + prob[10] + prob[11])
        spin.append(sp)
    del l

    for l in range(n_orbit):
        if(np.abs(enes[2*l]-enes[2*l + 1])<0.000001):
            spin[2*l] = 0
            spin[2*l+1] = 0
    del l

    return np.array(spin)

class CuO2:
    def __init__(self, Ne=2.0, k_mesh=31):
        """モデルのパラメータの設定

        Args:
            U (float): オンサイト相互作用の大きさ
            Ne (float, optional): 単位胞内での電子の数 Defaults to 6.0.
            k_mesh (int, optional): k点の細かさ Defaults to 31.
        """
        self.Ne         = Ne
        self.k_mesh     = k_mesh

        ne1 = Ne/12.0 + 0.2
        ne2 = Ne/12.0 - 0.2
        self.N_site_scf = np.array([
                        [ne1, ne1, ne2, ne2, ne2, ne2, ne1, ne1, ne2, ne2, ne2, ne2]])
        self.Ef_scf     = np.array([])
        self.Delta_scf  = np.array([0.8])
        self.Etot_scf   = np.array([0.8])

        self.delta      = 0
        self.ef         = 0
        self.enes       = np.zeros((k_mesh, k_mesh, n_orbit*2))
        self.eigenStates= np.zeros((k_mesh, k_mesh, n_orbit*2, n_orbit*2), dtype=np.complex128)
        self.spins      = np.zeros((k_mesh, k_mesh, n_orbit*2))

        self.path       = [("Γ", "Y"), ("Y", "M'"), ("M'", "Σ'"), ("Σ'","Γ"),
                           ("Γ", "Σ"), ("Σ", "M"), ("M", "X"), ("X", "Γ")]

        self.E          = 0
        self.dos        = np.array([])

        self.kF_index = np.array([[-1, -1, -1]])


    def calc_scf(self, iteration = 100, err = 1e-8):
        """自己無頓着計算を行う。delta と ef を決定する。

        Args:
            iteration (int, optional): 繰り返す回数の上限. Defaults to 100.
            err (float, optional): 収束条件. Defaults to 1e-6.
        """

        # 一度やったらもうやらない。
        if(self.Ef_scf.size > 1):
            print("SCF calculation was already done.")
            return

        print("SCF calculation start. Ne = {:1.2f}, err < {:1.1e}".format(self.Ne, err))

        kx, ky = self._gen_kmesh()

        # ここから自己無頓着方程式のループになる
        for scf_iteration in range(iteration):
            # Steffensen の反復法
            for m in range(2):
                # フェルミエネルギーを求める
                enes = []
                eigenEnes = np.zeros((self.k_mesh, self.k_mesh, n_orbit*2))
                eigenStates = np.zeros((self.k_mesh, self.k_mesh, n_orbit*2, n_orbit*2), dtype=np.complex128)
                Delta   = calc_delta(self.N_site_scf[-1])

                # ブリュアンゾーン内の全探査
                for i in range(self.k_mesh):
                    for j in range(self.k_mesh):
                        eigenEnergy, eigenState = Hamiltonian(kx[i][j],ky[i][j], Delta)
                        enes = np.append(enes, eigenEnergy)
                        eigenEnes[i,j] = eigenEnergy
                        eigenStates[i,j] = eigenState
                del i, j

                # 求めたエネルギー固有値をソートして下から何番目というのを探してやればよい
                # 絶縁体相のときのことを考えると平均をとる必要がある
                sorted_enes = np.sort(enes)
                ef = (sorted_enes[int(self.k_mesh * self.k_mesh * self.Ne) - 1]
                      + sorted_enes[int(self.k_mesh * self.k_mesh * self.Ne)])/2
                self.Ef_scf = np.append(self.Ef_scf, ef)

                # scf で求める値の初期化
                nsite  = np.zeros((n_orbit*2))
                etot   = 0

                # ブリュアンゾーン内の全探査
                for i in range(self.k_mesh):
                    for j in range(self.k_mesh):
                        # フェルミ分布で重みづけして足す。今は絶対零度なので階段関数になってる。
                        for l in range(n_orbit*2):
                            if (eigenEnes[i,j,l] <= ef) :
                                nsite += np.abs(eigenStates[i,j][:,l])**2
                                etot  += eigenEnes[i,j,l]
                del i, j, l

                # 規格化して足す
                nsite /= self.k_mesh * self.k_mesh
                self.N_site_scf = np.vstack((self.N_site_scf, nsite))

                self.Delta_scf = np.append(self.Delta_scf,calc_delta(nsite))

                etot /= self.k_mesh * self.k_mesh
                self.Etot_scf = np.append(self.Etot_scf, etot)


            del m

            # Steffensen の反復法
            ef = Steffensen(self.Ef_scf)
            self.Ef_scf = np.append(self.Ef_scf, ef)

            nsite = Steffensen(self.N_site_scf)
            self.N_site_scf = np.vstack((self.N_site_scf, nsite))

            delta = Steffensen(self.Delta_scf)
            self.Delta_scf = np.append(self.Delta_scf, delta)

            etot = Steffensen(self.Etot_scf)
            self.Etot_scf = np.append(self.Etot_scf, etot)


            # 与えられた誤差の範囲に収まったら終了する
            if(np.abs(self.Delta_scf[-1]-self.Delta_scf[-4]) < err) :

                self.delta = self.Delta_scf[-1]
                self.ef    = self.Ef_scf[-1]

                print("SCF loop converged.  Ne = {:1.2f}, err < {:1.1e}, loop = {:2d}, delta = {:1.2e}\n".format(self.Ne, err, scf_iteration*3, self.delta))

                return

        del scf_iteration

        # 収束しなかったときの処理
        self.delta = self.Delta_scf[-1]
        self.ef    = self.Ef_scf[-1]
        print('\033[41m'+"Calculation didn't converge. err > {:1.1e}, Ne = {:1.2f} loop = {:2d}, delta = {:1.2e}".format(err, self. Ne, iteration*3, self.delta)+'\033[0m')
        print(f"latter deltas are {self.Delta_scf[-4:-1]}\n")

        return


    def calc_nscf(self, fineness=5):
        """
        delta と ef が与えられたときの各k点の固有状態のエネルギー、状態ベクトル、スピンの大きさの計算をする
        """

        print("NSCF calculation start.")

        self.k_mesh *= fineness
        self.enes       = np.zeros((self.k_mesh, self.k_mesh, n_orbit*2))
        self.eigenStates= np.zeros((self.k_mesh, self.k_mesh, n_orbit*2, n_orbit*2), dtype=np.complex128)
        self.spins      = np.zeros((self.k_mesh, self.k_mesh, n_orbit*2))

        # ブリュアンゾーンのメッシュの生成
        kx, ky = self._gen_kmesh()

        # メッシュの各点でのエネルギー固有値の計算
        for i in range(self.k_mesh):
            for j in range(self.k_mesh):
                enes, eigenstate = Hamiltonian(kx[i][j],ky[i][j], self.delta)
                spin = calc_spin(enes, eigenstate)
                self.enes[i,j]         = enes
                self.eigenStates[i,j]  = eigenstate
                self.spins[i,j]        = np.array(spin)

        del i, j

        print("NSCF calculation finished.\n")
        return


    def calc_dos(self, E_fineness=1000, sigma2 = 0.0001):

        self.E = np.linspace(np.min(self.enes)-0.1, np.max(self.enes)+0.1, E_fineness)
        self.dos = np.array([])

        for e in self.E:
            self.dos = np.append(self.dos, np.sum(np.exp(-(e-self.enes)**2 / 2 / sigma2 ) / np.sqrt(2 * np.pi * sigma2)))
        del e

        self.dos /= np.sum(self.dos)*(self.E[1]-self.E[0])

        return


    def calc_kF_index(self):

        self.kF_index = np.array([[-1, -1, -1]])

        for i in range(self.k_mesh):
            for j in range(self.k_mesh):
                for m in range(n_orbit*2):
                    candidate_kF_index  = np.array([i, j, m])
                    ene_ij = self.enes[i,j,m]
                    # 八方で確かめる
                    if(i < self.k_mesh-1): # 南方向
                        ene_delta = self.enes[i+1,j,m]
                        if((ene_ij-self.ef)*(ene_delta-self.ef) < 0
                            and np.abs(ene_ij-self.ef) < np.abs(ene_delta-self.ef)):
                            self.kF_index = np.vstack((self.kF_index, candidate_kF_index))
                            continue

                    if(j < self.k_mesh-1): # 東方向
                        ene_delta = self.enes[i,j+1,m]
                        if((ene_ij-self.ef)*(ene_delta-self.ef) < 0
                            and np.abs(ene_ij-self.ef) < np.abs(ene_delta-self.ef)):
                            self.kF_index = np.vstack((self.kF_index, candidate_kF_index ))
                            continue

                    if(i > 0): # 北方向
                        ene_delta = self.enes[i-1,j,m]
                        if((ene_ij-self.ef)*(ene_delta-self.ef) < 0
                            and np.abs(ene_ij-self.ef) < np.abs(ene_delta-self.ef)):
                            self.kF_index = np.vstack((self.kF_index, candidate_kF_index ))
                            continue

                    if(j > 0): # 西方向
                        ene_delta = self.enes[i,j-1,m]
                        if((ene_ij-self.ef)*(ene_delta-self.ef) < 0
                            and np.abs(ene_ij-self.ef) < np.abs(ene_delta-self.ef)):
                            self.kF_index = np.vstack((self.kF_index, candidate_kF_index ))
                            continue

                    if(i < self.k_mesh-1 and j < self.k_mesh-1): # 南東方向
                        ene_delta = self.enes[i+1,j+1,m]
                        if((ene_ij-self.ef)*(ene_delta-self.ef) < 0
                            and np.abs(ene_ij-self.ef) < np.abs(ene_delta-self.ef)):
                            self.kF_index = np.vstack((self.kF_index, candidate_kF_index ))
                            continue

                    if(i > 0 and j < self.k_mesh-1): # 北東方向
                        ene_delta = self.enes[i-1,j+1,m]
                        if((ene_ij-self.ef)*(ene_delta-self.ef) < 0
                            and np.abs(ene_ij-self.ef) < np.abs(ene_delta-self.ef)):
                            self.kF_index = np.vstack((self.kF_index, candidate_kF_index))
                            continue

                    if(i > 0 and j > 0): # 北西方向
                        ene_delta = self.enes[i-1,j-1,m]
                        if((ene_ij-self.ef)*(ene_delta-self.ef) < 0
                            and np.abs(ene_ij-self.ef) < np.abs(ene_delta-self.ef)):
                            self.kF_index = np.vstack((self.kF_index, candidate_kF_index))
                            continue

                    if(i < self.k_mesh-1 and j > 0): # 南西方向
                        ene_delta = self.enes[i+1,j-1,m]
                        if((ene_ij-self.ef)*(ene_delta-self.ef) < 0
                            and np.abs(ene_ij-self.ef) < np.abs(ene_delta-self.ef)):
                            self.kF_index = np.vstack((self.kF_index, candidate_kF_index))
                            continue
        del i, j, m

        self.kF_index = np.delete(self.kF_index, 0, 0)
        return


    def calc_spin_conductivity(self, mu="x", nu="y", gamma=0.0001):
        if(self.enes[0,0,0] == 0):
            print("NSCF calculation wasn't done yet.")
            return

        print("SpinConductivity calculation start.")

        # フェルミ面の計算をしていなかったらする
        if(not hasattr(self, "kF_index")):
            self.calc_kF_index()

        # スピン伝導度 複素数として初期化
        chi = 0.0 + 0.0*1j

        # ブリュアンゾーンのメッシュの生成
        kx, ky = self._gen_kmesh()

        # バンド間遷移
        for i in range(self.k_mesh):
            for j in range(self.k_mesh):

                Jmu_matrix = np.conjugate(self.eigenStates[i,j].T) @ SpinCurrent(kx[i,j], ky[i,j], mu) @ self.eigenStates[i,j]
                Jnu_matrix = np.conjugate(self.eigenStates[i,j].T) @     Current(kx[i,j], ky[i,j], nu) @ self.eigenStates[i,j]

                for m in range(n_orbit*2):
                    for n in range(n_orbit*2):

                        Jmu = Jmu_matrix[m,n]
                        Jnu  = Jnu_matrix[n,m]

                        if(np.abs(self.enes[i,j,m]-self.enes[i,j,n]) > 1e-6):
                            # フェルミ分布
                            efm = 1 if (self.enes[i,j][m]<self.ef) else 0
                            efn = 1 if (self.enes[i,j][n]<self.ef) else 0

                            add_chi = Jmu * Jnu * (efm - efn) / ((self.enes[i,j][m]-self.enes[i,j][n])*(self.enes[i,j][m]-self.enes[i,j][n]+1j*gamma))
                            chi += add_chi
        del i, j, m, n

        # バンド内遷移
        for i, j, m in self.kF_index:

                Jmu_matrix = np.conjugate(self.eigenStates[i,j].T) @ SpinCurrent(kx[i,j], ky[i,j], mu) @ self.eigenStates[i,j]
                Jnu_matrix = np.conjugate(self.eigenStates[i,j].T) @     Current(kx[i,j], ky[i,j], nu) @ self.eigenStates[i,j]

                Jmu = Jmu_matrix[m,m]
                Jnu = Jnu_matrix[m,m]

                chi += 1j * Jmu * Jnu / gamma

        del i, j, m

        chi /= (self.k_mesh*self.k_mesh*1j)

        print("Spin Conductivity calculation finished")
        print("ReChi = {:1.2e}, ImChi = {:1.2e}\n".format(np.real(chi), np.imag(chi)))

        return chi


    def calc_conductivity(self, mu="x", nu="y", gamma=0.0001):
        if(self.enes[0,0,0] == 0):
            print("NSCF calculation wasn't done yet.")
            return

        print("Conductivity calculation start.")

        # フェルミ面の計算をしていなかったらする
        if(not hasattr(self, "kF_index")):
            self.calc_kF_index()

        # 伝導度 複素数として初期化
        sigma = 0.0 + 0.0*1j

        # ブリュアンゾーンのメッシュの生成
        kx, ky = self._gen_kmesh()

        # ブリュアンゾーンの和
        for i in range(self.k_mesh):
            for j in range(self.k_mesh):

                Jmu_matrix = np.conjugate(self.eigenStates[i,j].T) @  Current(kx[i,j], ky[i,j], mu) @ self.eigenStates[i,j]
                Jnu_matrix = np.conjugate(self.eigenStates[i,j].T) @  Current(kx[i,j], ky[i,j], nu) @ self.eigenStates[i,j]
                # 各波数におけるそれぞれの固有状態の和
                for m in range(n_orbit*2):
                    for n in range(n_orbit*2):
                        Jmu = Jmu_matrix[m,n]
                        Jnu = Jnu_matrix[n,m]

                        if(np.abs(self.enes[i,j,m]-self.enes[i,j,n]) > 1e-6):

                            # フェルミ分布
                            efm = 1 if (self.enes[i,j][m]<self.ef) else 0
                            efn = 1 if (self.enes[i,j][n]<self.ef) else 0

                            add_sigma = Jmu * Jnu * (efm - efn) / ((self.enes[i,j][m]-self.enes[i,j][n])*(self.enes[i,j][m]-self.enes[i,j][n]+1j*gamma))
                            sigma += add_sigma
        del i, j, m

        # バンド内遷移
        for i, j, m in self.kF_index:

                Jmu_matrix = np.conjugate(self.eigenStates[i,j].T) @  Current(kx[i,j], ky[i,j], mu) @ self.eigenStates[i,j]
                Jnu_matrix = np.conjugate(self.eigenStates[i,j].T) @  Current(kx[i,j], ky[i,j], nu) @ self.eigenStates[i,j]

                Jmu = Jmu_matrix[m,m]
                Jnu = Jnu_matrix[m,m]

                sigma += 1j * Jmu * Jnu / gamma
        del i, j, m

        sigma /= (self.k_mesh*self.k_mesh*1j)

        print("Conductivity calculation finished")
        print("ReSigma = {:1.2e}, ImSigma = {:1.2e}\n".format(np.real(sigma), np.imag(sigma)))

        return sigma


    def plot_nsite(self):
        if(self.Ef_scf.size < 2):
            print("SCF calculation wasn't done yet.")
            return

        plt.figure(figsize=[12.8,4.8])
        plt.subplot(121)
        for i in range(n_orbit):
            plt.plot(self.N_site_scf[:,i], label = "site {:d} = {:.3f}".format(i, self.N_site_scf[-1, i]))
        plt.legend()
        plt.subplot(122)
        for i in range(n_orbit,n_orbit*2):
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

        for kxy in k_path:
            enes, eigenstate = Hamiltonian(kxy[0], kxy[1],  self.delta)
            bands.append(enes)
            spin = calc_spin(enes, eigenstate)
            spins.append(spin)
        del kxy

        bands = np.array(bands)
        spins = np.array(spins)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
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

        for i in range(n_orbit*2):
            plt.scatter(distances, bands[:,i], c=spins[:,i]/2, cmap=cmap_name, vmin=-0.5, vmax=0.5, s=1)
        del i

        plt.vlines(label_loc[1:-1], Ymin,Ymax, "grey", "dashed")
        plt.hlines(self.ef, distances[0], distances[-1], "grey")
        plt.title("$E_f$ = {:.5f}".format(self.ef))
        plt.colorbar()

        plt.show()
        return


    def plot3d_band(self):
    # 参考 https://qiita.com/okumakito/items/3b2ccc9966c43a5e84d0

        if(self.Ef_scf.size < 2):
            print("SCF calculation wasn't done yet.")
            return

        kx, ky = self._gen_kmesh()

        fig = go.Figure()

        contours = dict(
            x=dict(highlight=False, show=True, color='grey', start=-3.5, end=3.5, size=0.5),
            y=dict(highlight=False, show=True, color='grey', start=-3.5, end=3.5, size=0.5),
            z=dict(highlight=False, show=False, start=-8, end = 8, size=0.5)
        )

        fig.add_trace(go.Surface(
                z=self.enes[:,:,0]-self.ef,
                x=kx,
                y=ky,
                surfacecolor=self.spins[:,:,0],
                colorscale = "viridis",
                cmin=-1.5,
                cmax=1.5,
                showscale = False,
                hoverinfo="skip",
                # opacity=0.8,
                # hidesurface=True,
            )
        )
        for i in range(1, n_orbit*2):
            fig.add_trace(go.Surface(
                    z=self.enes[:,:,i]-self.ef,
                    x=kx,
                    y=ky,
                    surfacecolor=self.spins[:,:,i],
                    colorscale = "viridis",
                    cmin=-1.5,
                    cmax=1.5,
                    showscale = False,
                    hoverinfo="skip",
                    contours=contours,
                    # opacity=0.8,
                    # hidesurface=True,
                )
            )
        del i

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
        return


    def plot_dos(self):
        if(self.dos.size < 2):
            self.calc_dos()

        E = np.linspace(np.min(self.enes)-0.1, np.max(self.enes)+0.1, self.dos.size)

        ysacale = np.max(self.dos)
        plt.ylim(-0.04*ysacale, 1.04*ysacale)

        plt.plot(E, self.dos)

        plt.xlabel("Energy (eV)")
        plt.ylabel("DOS")
        plt.vlines(self.ef, -0.04*ysacale, 1.04*ysacale, color="gray", linestyles="dashed")
        plt.title("Ef={:.2f} eV".format(self.ef))
        plt.show()

        return


    def plot_fermi_surface(self):
        if(self.kF_index.size < 4):
            self.calc_kF_index()

        kx, ky = self._gen_kmesh()

        for i, j, m in self.kF_index:
            color = "tab:green"
            if(self.spins[i,j,m] > 0.1):
                color = "tab:orange"
            if(self.spins[i,j,m] < -0.1):
                color = "tab:blue"
            plt.scatter(kx[i,j], ky[i,j], color=color, s=1)
        del i, j, m

        plt.axis("square")
        plt.xlim(-np.pi, np.pi)
        plt.ylim(-np.pi, np.pi)
        plt.show()
        return


    def _gen_kmesh(self, kx0 = 0.0, ky0 = 0.0, length = np.pi):
        kx = np.linspace(kx0 - length, kx0 + length, self.k_mesh)
        ky = np.linspace(ky0 - length, ky0 + length, self.k_mesh)
        return(np.meshgrid(kx, ky))
