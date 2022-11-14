import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from pathlib import Path

BASE_FOLDER = Path('/home/diegopc/projects/robot_formation_in_turbulence/results/ablation')
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

    # Ablation study
    ablation_folders = {
        'Ours - Full model': BASE_FOLDER / 'ours_k12',
        'Ours - No rel. position': BASE_FOLDER / 'no_rel',
        'Base MLP': BASE_FOLDER / 'mlp',
        'Wider MLP': BASE_FOLDER / 'mlp_wider',
        'Deeper MLP': BASE_FOLDER / 'mlp_deeper',
        'Only trajectory tracking': BASE_FOLDER / 'pd',
        #'ours k8': BASE_FOLDER / 'ours',
        #'ours k8 more r': BASE_FOLDER / 'ours_k8_more_r',
    }

    data_dict = {}
    for bs_name, bs_folder in ablation_folders.items():
        bs_data = np.load(bs_folder / 'eval_result_data.npy', allow_pickle=True).item()

        if len(data_dict) <= 0:
            for k, v in bs_data.items():
                data_dict[k] = v[None]
        else:
            for k, v in bs_data.items():
                data_dict[k] = np.concatenate([data_dict[k], v[None]], axis=0)

    for metric in ['position_error', 'velocity_error', 'velocity_dir_error']:
        data = data_dict[metric]
        num_bs, num_eval, num_robots, total_t = data.shape
        data = np.transpose(data[..., [6, 7, 8, 11, 12, 13, 16, 17, 18], :], (1, 2, 0, 3)).reshape(-1, num_bs, total_t)
        #data = np.transpose(data[..., 12, :], (1, 0, 2))[:15]

        if metric == 'position_error':
            plot_error(
                data, title=metric.capitalize().replace('_', ' '), name=metric, labels=list(ablation_folders.keys()),
                y_lim=[-0.02, 1.4], y_label='Error (m)',
                results_folder=BASE_FOLDER,
                legend_dict={'loc': 'upper left', 'size': 8, 'anchor': (0.01, 0.99)},
                colors=['red', 'blue', 'green', 'orange', 'brown', 'purple'],
            )

            table = pd.DataFrame(data.mean(axis=0)[:, [0, 150, 300, 450, 600, 750, 899]])
            table['method'] = list(ablation_folders.keys())
            with open(BASE_FOLDER / 'position_error_table.tex', 'w') as tb:
                tb.write(table.to_latex(index=False))

        elif metric == 'velocity_error':
            plot_error(
                data, title=metric.capitalize().replace('_', ' '), name=metric, labels=list(ablation_folders.keys()),
                y_lim=[-0.00, 0.22], y_label='Error (m/s)',
                results_folder=BASE_FOLDER,
                legend_dict={'loc': 'upper left', 'size': 8, 'anchor': (0.01, 0.99)},
                colors=['red', 'blue', 'green', 'orange', 'brown', 'purple'], smooth=True,
            )
        elif metric == 'velocity_dir_error':
            plot_error(
                np.arccos(-data + 1), title=metric.capitalize().replace('_', ' '), name=metric, labels=list(ablation_folders.keys()),
                y_lim=[-0.05, 1.05], y_label='Error (m)',
                results_folder=BASE_FOLDER,
                legend_dict={'loc': 'upper left', 'size': 8, 'anchor': (0.01, 0.99)},
                colors=['red', 'blue', 'green', 'orange', 'brown', 'purple'],
            )
