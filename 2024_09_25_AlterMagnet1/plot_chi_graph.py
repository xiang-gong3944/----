import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import funcs2 as fs
from funcs2 import KappaET2X
import os
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import cm  # For colormap
from matplotlib.colors import LinearSegmentedColormap

# Function to create and display the graph
def make_chi_graph(x, y, Ne, label):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlim(0.55, 1.45)
    ax1.hlines(0, 0.55, 1.45, color="grey")
    ax1.set_title(f"Ne = {Ne:.1f}")
    ax1.set_xlabel("U (eV)")
    ax1.set_ylabel(label)
    ax1.plot(x, y, label=f"{label} vs U")
    ax1.legend()

    # Save the graph as an image
    image_path = f"./output/{label}/"
    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    plt.savefig(image_path + f"Ne{int(Ne*10)}.png")

    # Display the graph
    #plt.show()

    return image_path



def make_3d_chi_graph(U_values, Ne_values, chi_values, label):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set white background and remove grid lines
    ax.set_facecolor('white')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White background for x-axis pane
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White background for y-axis pane
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White background for z-axis pane
    
    # Disable grid for all axes
    ax.xaxis._axinfo['grid'].update({
        'color': 'lightgrey',      # 他の部分のグリッドを保持する色
        'linewidth': 0.5,      # 線幅 
    })
    ax.yaxis._axinfo['grid'].update({
        'color': 'lightgrey',      # 他の部分のグリッドを保持する色
        'linewidth': 0.5,      # 線幅 
    })
    ax.zaxis._axinfo['grid'].update({
        'color': 'lightgrey',      # 他の部分のグリッドを保持する色
        'linewidth': 0.5,      # 線幅 
    })

    # gridのonとoff
    ax.grid(False)
    
    # y軸の各値に対応する平面を描画
    for y_val in np.arange(6.0, 7.0, 0.1):
        ax.plot([0.55, 1.45], [y_val, y_val], [0, 0], color='black', linewidth=0.5)
    # x軸の各値に対応する平面を描画
    for x_val in np.arange(0.6, 1.6, 0.2):
        ax.plot([x_val, x_val], [6.0, 6.9], [0, 0], color='lightgrey', linewidth=0.5)

    U_grid, Ne_grid = np.meshgrid(U_values, Ne_values)
    chi_grid = np.array(chi_values)

    # 色リストを定義
    color_list1 = ['dodgerblue', 'thistle']
    color_list2 = ['darkolivegreen', 'yellowgreen']
    # LinearSegmentedColormapで補間してカラーマップを作成
    custom_cmap1 = LinearSegmentedColormap.from_list('custom_cmap1', color_list1, N=256)
    custom_cmap2 = LinearSegmentedColormap.from_list('custom_cmap2', color_list2, N=256)
    # Normalize Ne values for colormap
    norm1 = plt.Normalize(vmin=min(Ne_values), vmax=max(Ne_values))
    norm2 = plt.Normalize(vmin=min(Ne_values), vmax=max(Ne_values))
    #norm2 = plt.Normalize(6.0, 6.2)

    for i, Ne in enumerate(Ne_values):
        chi_slice = chi_grid[i, :]
        for j in range(len(U_values) - 1):
            U_start, U_end = U_values[j], U_values[j + 1]
            chi_start, chi_end = chi_slice[j], chi_slice[j + 1]
            
            # Check if chi changes sign in this segment
            if chi_start * chi_end < 0:
                # Calculate the intersection point (linear interpolation)
                t = abs(chi_start) / (abs(chi_start) + abs(chi_end))
                U_intersect = U_start + t * (U_end - U_start)
                chi_intersect = 0  # At the intersection point

                # Split the segment into two parts at the intersection
                verts1 = [
                    [U_start, Ne, 0],
                    [U_start, Ne, chi_start],
                    [U_intersect, Ne, chi_intersect],
                    [U_intersect, Ne, 0]
                ]
                verts2 = [
                    [U_intersect, Ne, 0],
                    [U_intersect, Ne, chi_intersect],
                    [U_end, Ne, chi_end],
                    [U_end, Ne, 0]
                ]

                # Assign colors for each part
                color1 = custom_cmap1(norm1(Ne))  # Ne-based color
                color2 = custom_cmap1(norm1(Ne))  # Ne-based color

                # Add polygons
                poly1 = art3d.Poly3DCollection([verts1], color=color1, alpha=0.3, edgecolor='none')
                poly2 = art3d.Poly3DCollection([verts2], color=color2, alpha=0.3, edgecolor='none')
                ax.add_collection3d(poly1)
                ax.add_collection3d(poly2)
            elif chi_start * chi_end > 0:
                if chi_start < 0:
                    # No sign change, use a single segment
                    verts = [
                        [U_start, Ne, 0],
                        [U_start, Ne, chi_start],
                        [U_end, Ne, chi_end],
                        [U_end, Ne, 0]
                    ]
                    color = custom_cmap1(norm1(Ne))  # Ne-based color
                    poly = art3d.Poly3DCollection([verts], color=color, alpha=0.3, edgecolor='none')
                    ax.add_collection3d(poly)
                elif chi_start > 0:
                    # No sign change, use a single segment
                    verts = [
                        [U_start, Ne, 0],
                        [U_start, Ne, chi_start],
                        [U_end, Ne, chi_end],
                        [U_end, Ne, 0]
                    ]
                    color = custom_cmap2(norm2(Ne))  # Ne-based color
                    poly = art3d.Poly3DCollection([verts], color=color, alpha=0.3, edgecolor='none')
                    ax.add_collection3d(poly)

        # 各Neで線を描画（セグメントごとに正負を考慮）
        for j in range(len(U_values) - 1):
            U_start, U_end = U_values[j], U_values[j + 1]
            chi_start, chi_end = chi_slice[j], chi_slice[j + 1]
            
            # Check if chi changes sign
            if chi_start * chi_end < 0:
                # Calculate the intersection point
                t = abs(chi_start) / (abs(chi_start) + abs(chi_end))
                U_intersect = U_start + t * (U_end - U_start)
                chi_intersect = 0

                # Plot segments with different colors
                ax.plot([U_start, U_intersect], [Ne, Ne], [chi_start, chi_intersect], 
                        color=custom_cmap1(norm1(Ne)), linewidth=0.5)
                ax.plot([U_intersect, U_end], [Ne, Ne], [chi_intersect, chi_end], 
                        color=custom_cmap2(norm2(Ne)), linewidth=0.5)
            elif chi_start * chi_end > 0:
                if chi_start < 0:
                    # Plot a single segment
                    ax.plot([U_start, U_end], [Ne, Ne], [chi_start, chi_end], 
                            color=custom_cmap1(norm1(Ne)), linewidth=0.5)
                elif chi_start > 0:
                    ax.plot([U_start, U_end], [Ne, Ne], [chi_start, chi_end], 
                    color=custom_cmap2(norm2(Ne)), linewidth=0.5)


    # 軸ラベルとタイトル
    ax.set_xlabel("U (eV)")
    ax.set_ylabel("n")
    ax.set_zlabel("$\chi _{xy}/e$")
    ax.set_title("3D Plot of $\chi _{xy}$ vs U and Ne")
    ax.set_xlim(0.55, 1.45)
    ax.set_ylim(6.0, 6.9)
    # ax.set_zlim(-0.8, 0.4)

    # z軸chiを正負逆にする
    ax.invert_zaxis()

    # 横軸Neの目盛りを6.0から6.9まで0.1ずつ表示
    ax.set_yticks(np.arange(6.0, 6.9, 0.1))
    ax.set_xticks(np.arange(0.6, 1.45, 0.2))

    # 様々な角度から画像を保存
    for angle1 in range(-175, -125, 10):
        for angle2 in range(10, 40, 10):
            ax.view_init(elev=angle2, azim=angle1)  # elev: 仰角, azim: 方位角
            # Save the graph as an image
            image_path = f"./output/{label}/"
            if not os.path.isdir(image_path):
                os.makedirs(image_path)
            plt.savefig(image_path + f"chixy_3D_a{angle1}_e{angle2}.png", dpi=1500)

    plt.show()

    return image_path



# Main script
if __name__ == '__main__':
    # Model initialization
    Ne_values = np.linspace(6.0, 6.9, 10)
    U_values = np.linspace(0.6, 1.4, 19)
    
    chi_xy_all = []

    for Ne in Ne_values:
        chi_xy = []
        for U in U_values:
            model = KappaET2X(U, Ne, 51)
            model.calc_scf()
            model.calc_nscf()
            model.calc_kF_index()

            # Ensure `chi` matches the size of `x`
            mu, nu = "x", "y"
            chi = np.real(model.calc_spin_conductivity(mu, nu, gamma=0.0001))
            
            # Debug output
            print(f"U: {U}, Ne: {Ne}, chi: {chi}")

            chi_xy.append(chi)

        chi_xy_all.append(chi_xy)
        
        # Generate the graph for each Ne
        make_chi_graph(U_values, chi_xy, Ne, "Spin Conductivity")

    # Generate the 3D plot
    make_3d_chi_graph(U_values, Ne_values, chi_xy_all, "spin3D")