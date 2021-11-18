import matplotlib.pyplot as plt
import numpy as np
import math

""" get the most common unique states from a dataset """
# example: 
# unique_states,_,_ = get_most_common_unique_states(train_data,ndim_x=3,N=60,plot=True)
def get_most_common_unique_states(dataset,ndim_x=3,N=60,plot=False, save_fig_addr=None):
    # returns unique_states:
    # x[0], x[1], x[2], count, portion

    data_states = dataset[:,1:]
    unique_states,counts = np.unique(data_states, axis=0, return_counts=True)
    portions = counts/len(data_states)
    unique_states = np.append(unique_states, counts.reshape((len(counts),1)), axis=1)
    unique_states = np.append(unique_states, portions.reshape((len(portions),1)), axis=1)
    unique_states = sorted(unique_states,key=lambda l:l[-1], reverse=True)
    counts = np.sort(counts,axis=None)
    counts = counts[::-1]
    counts = counts.tolist()
    portions = np.sort(portions,axis=None)
    portions = portions[::-1]
    portions = portions.tolist()

    sum_counts = np.sum(counts[:N])
    sum_portions = np.sum(portions[:N])

    if plot is True:
        column_headers = ['x['+str(n)+']' for n in range(ndim_x)]
        column_headers.append('counts')
        column_headers.append('portion')

        row_headers = [str(n+1) for n in range(N)]
        cell_text = [[f'{item:3.4f}' for item in unique_state.tolist()] for unique_state in unique_states[0:N]]
        row_headers.append('sum')
        ent = ['-' for n in range(ndim_x)]
        ent.append(str(sum_counts))
        ent.append(f'{sum_portions:3.4f}')
        cell_text.append(ent)

        fig, ax =plt.subplots(figsize=(8,16))
        ax.set_axis_off()
        rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
        ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
        the_table = ax.table(cellText=cell_text,
                                    rowLabels=row_headers,
                                    rowColours=rcolors,
                                    rowLoc='right',
                                    colColours=ccolors,
                                    colLabels=column_headers,
                                    loc='center')

        ax.set_title("Total samples: "+ str(len(data_states)))

        if save_fig_addr is not None:
            plt.savefig(save_fig_addr+'common_states.png',bbox_inches='tight')

    return unique_states,sum_counts,sum_portions
