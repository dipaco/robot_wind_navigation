import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from pathlib import Path

BASE_FOLDER = Path('/home/diegopc/projects/robot_formation_in_turbulence/results/sensitivity/formation')
BOUNDS = [-5, 5]
DT = 0.066


def plot_error(data, title, labels, name, results_folder, y_label='', y_lim=None, legend_dict=None, colors=None, smooth=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(data.shape[-1]) * DT
    num_plots = data.shape[1]

    # colors = cm.rainbow(np.linspace(0, 1, eval_data.shape[0]))
    if colors is None:
        colors = cm.Set1(np.linspace(0, 1, data.shape[1]))
    #colors = cm.Dark2(np.linspace(0, 1, data.shape[1]))
    for i in range(num_plots):

        mean_data = data.mean(axis=0)
        sdt_data = data.std(axis=0)
        if smooth:
            #apply_along_axis(func1d, axis, arr, *args, **kwargs)
            pass

        #import pdb
        #pdb.set_trace()

        ax.plot(x, mean_data[i, :], label=labels[i], color=colors[i])
        ax.fill_between(x, mean_data[i] - sdt_data[i], mean_data[i] + sdt_data[i], color=colors[i],
                        alpha=0.1)

    if legend_dict is not None:
        #ax.legend(loc=legend_dict['loc'], bbox_to_anchor=(1.1, 1.05), prop={'size': legend_dict['size']})
        ax.legend(loc=legend_dict['loc'], bbox_to_anchor=legend_dict['anchor'], prop={'size': legend_dict['size']})
    ax.grid()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(y_label)
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_title(f'{title}')
    fig.savefig(results_folder / f'{name}.png')
    plt.close(fig)

    # robot error figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.boxplot(np.transpose(data, axes=[0, 2, 1]).reshape(-1, num_plots))

    ax.legend(loc='upper right', prop={'size': 6})
    ax.grid()
    ax.set_ylim(0.05 * BOUNDS[0], 0.3 * BOUNDS[1])
    ax.set_title(f'{title} per robot')
    fig.savefig(results_folder / f'{name}_per_robot.png')
    plt.close(fig)


if __name__ == '__main__':

    # Train test folders

    formations = list(range(3, 9))

    formation_folders = [[BASE_FOLDER / f'train_{i}x{i}_test_{j}x{j}' for j in formations] for i in formations]

    data_dict = {k: [] for k in np.load(formation_folders[0][0] / 'eval_result_data.npy', allow_pickle=True).item().keys()}
    for i in range(len(formations)):

        for k, v in data_dict.items():
            data_dict[k].append([])

        for j in range(len(formations)):
            bs_folder = formation_folders[i][j]
            bs_data = np.load(bs_folder / 'eval_result_data.npy', allow_pickle=True).item()

            for k, v in bs_data.items():
                data_dict[k][i].append(v)

        if i >= 1:
            break

    for metric in ['position_error', 'velocity_error', 'velocity_dir_error']:
        data = data_dict[metric]

        table = np.zeros((len(formations), len(formations)))
        for i in range(len(formations)):
            for j in range(len(formations)):
                table[i, j] = data[i][j].mean()

            if i >= 1:
                break

        table = pd.DataFrame(table)
        table['Training team size'] = np.array(formations)**2

        with open(BASE_FOLDER / 'training_test.tex', 'w') as tb:
            tb.write(table.to_latex(index=False))
        import pdb
        pdb.set_trace()

        num_bs, num_eval, num_robots, total_t = data.shape
        data = np.transpose(data[..., [6, 7, 8, 11, 12, 13, 16, 17, 18], :], (1, 2, 0, 3)).reshape(-1, num_bs, total_t)
        #data = np.transpose(data[..., 12, :], (1, 0, 2))[:15]

        if metric == 'position_error':
            '''plot_error(
                data, title=metric.capitalize().replace('_', ' '), name=metric, labels=list(ablation_folders.keys()),
                y_lim=[-0.02, 1.4], y_label='Error (m)',
                results_folder=BASE_FOLDER,
                legend_dict={'loc': 'upper left', 'size': 8, 'anchor': (0.01, 0.99)},
                colors=['red', 'blue', 'green', 'orange', 'brown', 'purple'],
            )'''


            #table = pd.DataFrame(data.mean(axis=0)[:, [0, 150, 300, 450, 600, 750, 899]])

