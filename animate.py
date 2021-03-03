import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def read_npz(file_path):
    data = np.load(file_path, allow_pickle=True)
    data = data['arr_0'].reshape(1,-1)[0][0]
    total_nframe = len(data.keys())
    return total_nframe, data



def main():
    fig = plt.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=15., azim=0)
    lines = []

    def update(i, kpts_stack, fig):
        nonlocal lines
        joint_pairs = (
            (0,1), (1,14), #spine
            (2,5), #shoulder
            (2,3), (3,4), #L_arm
            (5,6), (6,7), #R_arm
            (0,8), (8,9), (9,10), #L_leg
            (0,11), (11,12), (11,13) #R_leg
        )
        kpt = kpts_stack[i][0,0,:,:].cpu().detach().numpy()
        assert kpt.shape == (15, 3)

        
        for pair in joint_pairs:
            X = (kpt[pair[0], 0], kpt[pair[1], 0])
            Y = (kpt[pair[0], 1], kpt[pair[1], 1])
            Z = (kpt[pair[0], 2], kpt[pair[1], 2])
            plt.plot(X,Y,Z, zdir='z')

    nframe, kpts_stack = read_npz("output_pts.npz")
    anim = FuncAnimation(fig, update, frames=nframe, fargs=(kpts_stack, fig))
    plt.draw()
    plt.show()


if __name__ == "__main__":
    main()