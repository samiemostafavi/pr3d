import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os
import math
import bisect
import statistics
import matplotlib.ticker as mticker

from cde.data_collector import MatlabDataset, MatlabDatasetH5, get_most_common_unique_states
from cde.density_estimator import plot_conditional_hist, measure_percentile, measure_percentile_allsame, measure_tail, measure_tail_allsame, init_tail_index_hill, estimate_tail_index_hill

""" Empirical Data Analysis """

def evaluate_models_singlestate(models,model_names,train_data,cond_state=[0,1,7],file_addr='../../data/cond_records_[0_1_7]_92M.mat',quantiles=[0.8,1-1e-1,1-1e-3,1-1e-5,1-1e-7],test_dataset=None):
    
    n_models = len(models)

    if test_dataset is None:
        #matds = MatlabDatasetH5('../../data/cond_records_[0_1_7]_92M.mat') 
        cond_matds = MatlabDatasetH5(file_addr)
        #cond_state = [0,1,7]
        test_data = cond_matds.get_data(cond_matds.n_records)
    else:
        test_data = np.squeeze(test_dataset[np.where((test_dataset[:,1:]==cond_state).all(axis=1)),:])

    #fig, ax = plt.subplots(figsize=(9*2,6+3))
    #fig, axs = plt.subplots(nrows=2, ncols=2,figsize=(9*2,6+3))
    fig, axs = plt.subplots(3, 2, figsize=(9*2,6+3+3), gridspec_kw={'height_ratios': [2, 1, 1]})


    # create x
    measured_p0,num_samples_test,avg = measure_percentile_allsame(dataset=test_data,p_perc=0)
    measured_p8,num_samples_test,avg = measure_percentile_allsame(dataset=test_data,p_perc=80)
    measured_p1,num_samples_test,avg = measure_percentile_allsame(dataset=test_data,p_perc=100)
    width = 0.1
    xlim = [np.floor(measured_p0),np.ceil(measured_p1+1),np.ceil(measured_p8),np.ceil(measured_p1+1)]
    if(xlim[0]<=0.1):
        xlim[0] = 0.1
    x = np.arange(start=xlim[0], stop=xlim[1], step=width)
    x_edges = x-(width/2)
    x_edges = np.append(x_edges,x[-1]+(width/2))

    # train data histogram
    try:
        tt,num_samples_train,avg = measure_percentile(dataset=train_data,x_cond=np.array([cond_state]),p_perc=100)
        edataset = train_data
        conditioned_ds = edataset[np.where(np.all(edataset[:,1:]==cond_state,axis=1))]
        conditioned_ds = conditioned_ds[:,0]
        train_hist, bin_edges = np.histogram(conditioned_ds, bins=x_edges, density=True)
        axs[0,0].plot(x,train_hist, marker='.', label="empirical train "+str(num_samples_train), linestyle = ':')
    except:
        num_samples_train = 0

    #x = np.logspace(math.log10( measured_p8 ), math.log10( measured_1+1), num=60)

    # test data histogram
    edataset = test_data
    conditioned_ds = edataset[np.where(np.all(edataset[:,1:]==cond_state,axis=1))]
    conditioned_ds = conditioned_ds[:,0]
    test_hist, bin_edges = np.histogram(conditioned_ds, bins=x_edges, density=True)
    axs[0,0].plot(x,test_hist, marker='.', label="empirical histogram samples: "+str(num_samples_test), linestyle = ':')
    hist_mean = np.sum(x*test_hist*width)
    #print("empirical mean: " + str(np.sum(x*test_hist*width)))

    # models pdf
    model_means = []
    for j in range(len(models)):
        prob=[]
        for i in range(len(x)):
            mx = np.array([cond_state])
            my = np.array([x[i]])
            prob.append(models[j].pdf(mx,my))
        
        plabel = model_names[j]
        try:
            tail_threshold, tail_param = models[j]._get_tail_components(mx)
            tail_threshold = np.squeeze(tail_threshold)
            plabel = plabel + " threshold=" + str(tail_threshold)
        except:
            pass

        axs[0,0].plot(x,prob, label=plabel)
        model_means.append(np.sum(x*np.squeeze(prob)*width))
        #print("emm mean: " + str(np.sum(x*np.squeeze(prob_emm)*width)))

    axs[0,0].set_xlabel('latency [time]')
    #ax.set_xticks([5,6,7,8,9,10])
    #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs[0,0].set_ylabel('probability')
    axs[0,0].set_title("state: " +str(cond_state)+ " training samples: "+ str(num_samples_train))
    axs[0,0].legend()
    axs[0,0].grid()

    # tail probability
    x = np.logspace(math.log10( xlim[2] ), math.log10( xlim[3] ), num=60)

    # train data tail
    if(num_samples_train != 0):
        train_tail=[]
        for i in range(len(x)):
            train_tail.append(measure_tail(dataset=train_data,x_cond=np.array([cond_state]),y=x[i]))
        axs[0,1].loglog(x,train_tail, marker='.', label="empirical train "+str(num_samples_train), linestyle = 'None') 

    # test data tail
    testd_sorted = np.sort(test_data[:,0])
    test_tail=[]
    for i in range(len(x)):
        indx = bisect.bisect_left(testd_sorted, x[i])
        test_tail.append((len(test_data)-indx)/len(test_data))
    axs[0,1].loglog(x,test_tail, marker='.', label="empirical test "+str(num_samples_test), linestyle = 'None')

    # models tail
    model_tails = []
    for j in range(len(models)):
        tail=[]
        for i in range(len(x)):
            mx = np.array([cond_state])
            my = np.array([x[i]])
            #try:
            tail.append(models[j].tail(mx,my))
            #except:
            #    tail.append(models[j].find_tail(x_cond=mx,y=x[i],init_bound=200))
        
        plabel = model_names[j]
        try:
            tail_threshold, tail_param = models[j]._get_tail_components(mx)
            tail_threshold = np.squeeze(tail_threshold)
            plabel = plabel + " threshold=" + str(tail_threshold)
        except:
            pass

        axs[0,1].loglog(x,tail, label=plabel)
        model_tails.append(tail)

    axs[0,1].set_xlabel('latency [log]')
    axs[0,1].set_ylabel('Tail probability [log]')
    axs[0,1].set_title("state: " +str(cond_state)+ " training samples: "+ str(num_samples_train))
    axs[0,1].legend()
    axs[0,1].grid()

    # table

    test_quants = []
    num_sample_quants = []
    for quant in quantiles:
        test_perc,_,_ = measure_percentile_allsame(dataset=test_data,p_perc=quant*100)
        num_sampletest = len(test_data[test_data>=test_perc])
        test_quants.append(test_perc)
        num_sample_quants.append(num_sampletest)

    row_headers = ['empirical','emp samples']
    column_headers = ['mean']
    cell_text = [[f'{hist_mean:3.3f}']+['']*len(test_quants),
                 [str(num_samples_test)]+['']*len(test_quants)]

    model_quants = []
    for i in range(len(models)):
        quants = []
        cstate = np.array([cond_state])
        for quant in quantiles:
            quants.append(models[i].find_perc(alpha=1-quant,eps=1e-2,x_cond=cstate,init_bound=200))
        row_headers.append(model_names[i])
        cell_text.append([f'{model_means[i]:3.3f}']+['']*len(test_quants))
        model_quants.append(quants)

    for i in range(len(test_quants)):
        column_headers.append(str(quantiles[i]))
        cell_text[0][i+1] = f'{test_quants[i]:3.3f}'
        cell_text[1][i+1] = str(num_sample_quants[i])
        for j in range(len(models)):
            cell_text[2+j][i+1] = f'{model_quants[j][i]:3.3f}'
            cell_text[2+j][i+1] = f'{model_quants[j][i]:3.3f}'

    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    the_table = axs[1,0].table(cellText=cell_text,
                                rowLabels=row_headers,
                                rowColours=rcolors,
                                rowLoc='right',
                                colColours=ccolors,
                                colLabels=column_headers,
                                loc='center')


    # Hide table axes
    axs[1,0].get_xaxis().set_visible(False)
    axs[1,0].get_yaxis().set_visible(False)
    # Hide table axes border
    axs[1,0].spines["top"].set_visible(False)
    axs[1,0].spines["right"].set_visible(False)
    axs[1,0].spines["left"].set_visible(False)
    axs[1,0].spines["bottom"].set_visible(False)


    # Tail error plot
    for i in range(len(models)):
        newt = np.array(model_tails[i][0:len(test_tail)])
        model_error = np.log10(newt)-np.log10(test_tail)
        axs[1,1].plot(x,model_error, label=model_names[i])

    axs[1,1].set_xscale('log')
    axs[1,1].set_xlabel('latency [log]')
    axs[1,1].set_ylabel('Tail probability error [log]')
    axs[1,1].legend()
    axs[1,1].grid()

    # Quantiles bar plot
    labels = column_headers
    xbar = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    each_bar_width = width/len(models)
    each_bar_start = xbar - width/2
    for j in range(len(models)):
        model_norm = [model_means[j]/hist_mean]
        for i in range(len(test_quants)):
            model_norm.append(model_quants[j][i]/test_quants[i])
    
        axs[2,0].bar(each_bar_start + j*each_bar_width, model_norm, each_bar_width, label=model_names[j])

    xhl = np.arange(len(labels)+2)-1
    h_line = [1 for _ in xhl]
    axs[2,0].plot(xhl,h_line, label='Target')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[2,0].set_ylabel('standardized quantiles')
    axs[2,0].set_title('Standardized estimated quantiles')
    axs[2,0].set_xticks(xbar)
    axs[2,0].set_xticklabels(labels)
    axs[2,0].legend()

    #axs[2,0].bar_label(rects1, padding=3)
    #axs[2,0].bar_label(rects2, padding=3)

    # delete the remaining figure
    fig.delaxes(axs[2,1])

    plt.show()

""" Empirical Data Analysis """

def evaluate_models_save_plots(models,model_names,train_data,cond_state=[0,1,7],file_addr='../../data/cond_records_[0_1_7]_92M.mat',quantiles=[0.8,1-1e-1,1-1e-3,1-1e-5,1-1e-7],test_dataset=None,save_fig_addr=None):
    
    n_models = len(models)

    if test_dataset is None:
        #matds = MatlabDatasetH5('../../data/cond_records_[0_1_7]_92M.mat') 
        cond_matds = MatlabDatasetH5(file_addr)
        #cond_state = [0,1,7]
        test_data = cond_matds.get_data(cond_matds.n_records)
    else:
        test_data = np.squeeze(test_dataset[np.where((test_dataset[:,1:]==cond_state).all(axis=1)),:])

    fig, ax = plt.subplots(figsize=(10,5))
    #fig, axs = plt.subplots(nrows=2, ncols=2,figsize=(9*2,6+3))
    #fig, axs = plt.subplots(3, 2, figsize=(9*2,6+3+3), gridspec_kw={'height_ratios': [2, 1, 1]})

    # create x
    measured_p0,num_samples_test,avg = measure_percentile_allsame(dataset=test_data,p_perc=0)
    measured_p8,num_samples_test,avg = measure_percentile_allsame(dataset=test_data,p_perc=80)
    measured_p1,num_samples_test,avg = measure_percentile_allsame(dataset=test_data,p_perc=99.999)
    width = 0.1
    xlim = [np.floor(measured_p0),np.ceil(measured_p1),np.ceil(measured_p8),np.ceil(measured_p1)]
    #if(xlim[0]<=0.1):
    #    xlim[0] = 0.1
    x = np.arange(start=xlim[0], stop=xlim[1], step=width)
    x_edges = x-(width/2)
    x_edges = np.append(x_edges,x[-1]+(width/2))

    # train data histogram
    try:
        tt,num_samples_train,avg = measure_percentile(dataset=train_data,x_cond=np.array([cond_state]),p_perc=100)
        edataset = train_data
        conditioned_ds = edataset[np.where(np.all(edataset[:,1:]==cond_state,axis=1))]
        conditioned_ds = conditioned_ds[:,0]
        train_hist, bin_edges = np.histogram(conditioned_ds, bins=x_edges, density=True)
        ax.plot(x,train_hist, marker='.', label="Training data "+str(num_samples_train)+" samples", linestyle = ':')
    except:
        num_samples_train = 0

    #x = np.logspace(math.log10( measured_p8 ), math.log10( measured_1+1), num=60)

    # test data histogram
    edataset = test_data
    conditioned_ds = edataset[np.where(np.all(edataset[:,1:]==cond_state,axis=1))]
    conditioned_ds = conditioned_ds[:,0]
    test_hist, bin_edges = np.histogram(conditioned_ds, bins=x_edges, density=True)
    ax.plot(x,test_hist, marker='.', label="Test data "+str(num_samples_test)+" samples", linestyle = ':')
    hist_mean = np.sum(x*test_hist*width)
    #print("empirical mean: " + str(np.sum(x*test_hist*width)))

    # models pdf
    model_means = []
    for j in range(len(models)):
        prob=[]
        for i in range(len(x)):
            mx = np.array([cond_state])
            my = np.array([x[i]])
            prob.append(models[j].pdf(mx,my))
        
        plabel = model_names[j]
        try:
            tail_threshold, tail_param = models[j]._get_tail_components(mx)
            tail_threshold = np.squeeze(tail_threshold)
            plabel = plabel + " threshold=" + str(tail_threshold)
        except:
            pass

        ax.plot(x,prob, label=plabel)
        model_means.append(np.sum(x*np.squeeze(prob)*width))
        #print("emm mean: " + str(np.sum(x*np.squeeze(prob_emm)*width)))

    #ax.set_xlim([xlim[0], xlim[1]])
    ax.set_xlabel('Latency')
    #ax.set_xticks([5,6,7,8,9,10])
    #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel('Probability')
    #ax.set_title("state: " +str(cond_state)+ " training samples: "+ str(num_samples_train))
    ax.legend()
    ax.grid()

    if save_fig_addr is not None:
        plt.savefig(save_fig_addr+'fig1_state'+str(cond_state)+'.png',bbox_inches='tight')
    else:
        plt.show()

    fig, ax = plt.subplots(figsize=(10,5))

    # tail probability
    x = np.logspace(math.log10( xlim[2] ), math.log10( xlim[3] ), num=60)

    # train data tail
    if(num_samples_train != 0):
        train_tail=[]
        for i in range(len(x)):
            train_tail.append(measure_tail(dataset=train_data,x_cond=np.array([cond_state]),y=x[i]))
        ax.loglog(x,train_tail, marker='.', label="Training data "+str(num_samples_train)+" samples", linestyle = 'None') 

    # test data tail
    testd_sorted = np.sort(test_data[:,0])
    test_tail=[]
    for i in range(len(x)):
        indx = bisect.bisect_left(testd_sorted, x[i])
        test_tail.append((len(test_data)-indx)/len(test_data))
    ax.loglog(x,test_tail, marker='.', label="Test data "+str(num_samples_test)+" samples", linestyle = 'None')

    # models tail
    model_tails = []
    for j in range(len(models)):
        tail=[]
        for i in range(len(x)):
            mx = np.array([cond_state])
            my = np.array([x[i]])
            #try:
            tail.append(models[j].tail(mx,my))
            #except:
            #    tail.append(models[j].find_tail(x_cond=mx,y=x[i],init_bound=200))
        
        plabel = model_names[j]
        try:
            tail_threshold, tail_param = models[j]._get_tail_components(mx)
            tail_threshold = np.squeeze(tail_threshold)
            plabel = plabel + " threshold=" + str(tail_threshold)
        except:
            pass

        ax.loglog(x,tail, label=plabel)
        model_tails.append(tail)

    ax.set_ylim([1e-8, 1])
    #ax.set_xticks([6,10,15,20])
    ax.set_xticks(range(math.ceil(xlim[2]),math.floor(xlim[3])+1,3))
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Latency [log]')
    ax.set_ylabel('Tail probability [log]')
    #ax.set_title("state: " +str(cond_state)+ " training samples: "+ str(num_samples_train))
    ax.legend()
    ax.grid()

    if save_fig_addr is not None:
        plt.savefig(save_fig_addr+'fig2_state'+str(cond_state)+'.png',bbox_inches='tight')
    else:
        plt.show()

    # table

    fig, ax = plt.subplots(figsize=(8,4))

    test_quants = []
    num_sample_quants = []
    for quant in quantiles:
        test_perc,_,_ = measure_percentile_allsame(dataset=test_data,p_perc=quant*100)
        num_sampletest = len(test_data[test_data>=test_perc])
        test_quants.append(test_perc)
        num_sample_quants.append(num_sampletest)

    row_headers = ['empirical','emp samples']
    column_headers = ['mean']
    cell_text = [[f'{hist_mean:3.3f}']+['']*len(test_quants),
                 [str(num_samples_test)]+['']*len(test_quants)]

    model_quants = []
    for i in range(len(models)):
        quants = []
        cstate = np.array([cond_state])
        for quant in quantiles:
            quants.append(models[i].find_perc(alpha=1-quant,eps=1e-2,x_cond=cstate,init_bound=200))
        row_headers.append(model_names[i])
        cell_text.append([f'{model_means[i]:3.3f}']+['']*len(test_quants))
        model_quants.append(quants)

    for i in range(len(test_quants)):
        column_headers.append(str(quantiles[i]))
        cell_text[0][i+1] = f'{test_quants[i]:3.3f}'
        cell_text[1][i+1] = str(num_sample_quants[i])
        for j in range(len(models)):
            cell_text[2+j][i+1] = f'{model_quants[j][i]:3.3f}'
            cell_text[2+j][i+1] = f'{model_quants[j][i]:3.3f}'

    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    the_table = ax.table(cellText=cell_text,
                                rowLabels=row_headers,
                                rowColours=rcolors,
                                rowLoc='right',
                                colColours=ccolors,
                                colLabels=column_headers,
                                loc='center')


    # Hide table axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide table axes border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    if save_fig_addr is not None:
        plt.savefig(save_fig_addr+'fig3_state'+str(cond_state)+'.png',bbox_inches='tight')
    else:
        plt.show()


    fig, ax = plt.subplots(figsize=(8,4))

    # Tail error plot
    for i in range(len(models)):
        newt = np.array(model_tails[i][0:len(test_tail)])
        model_error = np.log10(newt)-np.log10(test_tail)
        ax.plot(x,model_error, label=model_names[i])

    ax.set_xscale('log')
    ax.set_xlabel('latency [log]')
    ax.set_ylabel('Tail probability error [log]')
    ax.legend()
    ax.grid()


    if save_fig_addr is not None:
        plt.savefig(save_fig_addr+'fig4_state'+str(cond_state)+'.png',bbox_inches='tight')
    else:
        plt.show()


    fig, ax = plt.subplots(figsize=(8,4))

    # Quantiles bar plot
    labels = column_headers
    xbar = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    each_bar_width = width/len(models)
    each_bar_start = xbar - width/2
    for j in range(len(models)):
        model_norm = [model_means[j]/hist_mean]
        for i in range(len(test_quants)):
            model_norm.append(model_quants[j][i]/test_quants[i])
    
        ax.bar(each_bar_start + j*each_bar_width, model_norm, each_bar_width, label=model_names[j])

    xhl = np.arange(len(labels)+2)-1
    h_line = [1 for _ in xhl]
    ax.plot(xhl,h_line, label='Target')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('standardized quantiles')
    ax.set_title('Standardized estimated quantiles')
    ax.set_xticks(xbar)
    ax.set_xticklabels(labels)
    ax.legend()

    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)


    if save_fig_addr is not None:
        plt.savefig(save_fig_addr+'fig5_state'+str(cond_state)+'.png',bbox_inches='tight')
    else:
        plt.show()



class empirical_measurer():
    def __init__(self, dataset, xsize=3,quantiles=[0.8,0.9,0.99]):
        # x_cond | mean | quantiles | number of samples

        self.database = np.empty((0,xsize+1+len(quantiles)+1))
        self.dataset = dataset
        self.quantiles = quantiles
        self.xsize = xsize

    def query_database(self,x_cond):
        if self.database.size is 0:
            return False, np.array([])
        condition = (self.database[:,:self.xsize]==x_cond)
        index = np.where(condition.all(axis=1))
        if len(index[0]) is 0:
            return False, np.array([])
        else:
            index = np.squeeze(index)
            return True, self.database[index]

    def measure_quantiles(self,x_cond):
        result, entry = self.query_database(x_cond)
        if result is False:
            # calculate the quantiles and mean
            conditioned_ds = np.squeeze(self.dataset[np.where((self.dataset[:,1:]==x_cond).all(axis=1)),0])
            measured_quants = np.quantile(conditioned_ds, self.quantiles)
            mean = np.mean(conditioned_ds)
            # insert the entry to the database
            # x_cond | mean | quantiles | number of samples
            entry = np.concatenate((x_cond,[mean],measured_quants,[len(conditioned_ds)]))
            self.database = np.append(self.database,[entry],axis=0)
        
        return entry[self.xsize:self.xsize+len(self.quantiles)+1],entry[-1]

""" empirical_measurer example """
"""
start_time = time.time()
em = empirical_measurer(test_data,3,[0.8,0.9,0.99])
# array(mean,quantiles), number of samples
print(em.measure_quantiles(x_cond=[0,2,7]))
elapsed_time = time.time() - start_time
print(f'{elapsed_time:3.1f}')

(array([ 9.39162425, 10.50903232, 11.04100028, 13.07094351]), 1711.0)
51.6
"""


def evaluate_model_allstates(emp_model,model,train_data,unique_states,N=60,quantiles=[0.9, 0.99, 0.999, 0.9999, 0.99999],xsize=3,root_find=True):
    # the first model treats with .tail_inverse, the rest with find_perc
    # N=60, len(quantiles)=5, 1.5 hour

    unique_states_np = np.array(unique_states)

    train_len = len(train_data)
  
    results = np.empty((0,xsize+(len(quantiles))+1)) # without mean
    for n in range(N):
        # check the system state
        x_n = unique_states_np[n,:xsize]
        #print(x_n)

        # calculate quantiles from empirical data
        c_n, test_samples_n = emp_model.measure_quantiles(x_cond=x_n)
        # no mean
        c_n = c_n[1:]
        #print(c_n)

        cstate = np.array([x_n])
        ch_n = np.array([])
        for quant in quantiles:
            #ch_emm_n = np.append(ch_emm_n,model.find_perc(alpha=1-quant,eps=1e-2,x_cond=cstate,init_bound=200))
            if root_find is False:
                ch_n = np.append(ch_n,model.tail_inverse(X=cstate,T=1-quant,init_bound=200,eps=1e-2))
            else:
                ch_n = np.append(ch_n,model.find_perc(alpha=1-quant,eps=1e-2,x_cond=cstate,init_bound=200))
    
        #print(ch_n)
        entry = np.concatenate((x_n,(ch_n-c_n)/c_n,[test_samples_n/len(emp_model.dataset)]))
        #print(entry)

        #print(test_samples_n)
        results = np.append(results,[entry],axis=0)
    
    return results

""" empirical_measurer example """
"""
start_time = time.time()
em = empirical_measurer(test_data,3,[0.8,0.9,0.99])
# array(mean,quantiles), number of samples
print(em.measure_quantiles(x_cond=[0,2,7]))
elapsed_time = time.time() - start_time
print(f'{elapsed_time:3.1f}')

(array([ 9.39162425, 10.50903232, 11.04100028, 13.07094351]), 1711.0)
51.6
"""


def evaluate_model_allstates_tail(models,test_data,unique_states,N=20,N_y=20,ylim=[0.00001, 0.99999],xsize=3):
    # the first model treats with .tail_inverse, the rest with find_perc
    # N=60, len(quantiles)=5, 1.5 hour

    unique_states_np = np.array(unique_states)

    x_quants = [1-i for i in np.logspace(math.log10( ylim[1] ), math.log10( ylim[0] ) , num=N_y)]
    x_axis = []
    for n in range(N):
        cond_state = unique_states_np[n,:xsize]
        x = []
        for i in range(len(x_quants)):
            dataset_perc,num_samples_train,avg = measure_percentile(dataset=test_data,x_cond=np.array([cond_state]),p_perc=x_quants[i]*100)
            x.append(dataset_perc)
        
        x_axis.append(x)
        #x = np.arange(start=ylim[0], stop=ylim[1], step=(ylim[1]-ylim[0])/N_y)

    # iterate over the unique states
    results = []

    # check the model
    for j in range(len(models)):
        
        model_tail_errors=[]
        model = models[j]
        for n in range(N):
            # bring the 
            x = x_axis[n]
            # check the system state
            cond_state = unique_states_np[n,:xsize]

            # quantile of test_tail
            test_tail=np.ones(len(x_quants))-x_quants

            tail=[]
            for i in range(len(x)):
                mx = np.array([cond_state])
                my = np.array([x[i]])
                res = model.tail(mx,my)
                if res < 1e-16:
                    res = 1e-16
                tail.append(res)

            # Model Tail error
            us_tail_errors = abs(np.log10(tail)-np.log10(test_tail))
            model_tail_errors.append(us_tail_errors)
        
        results.append(np.mean(model_tail_errors, axis=0))

    return results,x_axis,x_quants


def evaluate_models_allstates_plot(cma_results,train_len,model_names=['EMM-GPD','GMM'],quantiles=[0.9, 0.99, 0.999, 0.9999, 0.99999],xsize=3,markers=['s','o','v','*','x'],loglog=False, ylim_ll=[1e-4,1e0],ylim=[0,0.4]):
    
    n_plots = len(cma_results)
    fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots,4))

    for m in range(n_plots):
        results = cma_results[m]
        # start the plot
        leg_custs = []
        if n_plots is 1:
            axe = axes
        else:
            axe = axes[m]

        ax0t = axe.twiny()
        for i in range(len(quantiles)):
            colors_arr = np.where(results[:,xsize+i] > 0, 'green', 'red')
            path = axe.scatter(results[:,-1],abs(results[:,xsize+i]),c=colors_arr,marker=markers[i])
            # custom legend labels and markers
            leg_custs.append(mlines.Line2D([], [], color='black', marker=markers[i], linestyle='None',markersize=6, label=str(quantiles[i])))

        red_patch = mpatches.Patch(color='red', label='error<0')
        green_patch = mpatches.Patch(color='green', label='error>0')
        leg_custs.append(red_patch)
        leg_custs.append(green_patch)
        axe.legend(handles=leg_custs)
        axe.grid()
        axe.set_title(model_names[m])

        if loglog is False:
            axe.set_ylim(ylim)
            axe.set_ylabel('Standardized error of quantiles')
            axe.set_xlabel('P(x)')
        else:
            axe.set_ylim(ylim_ll)
            ax0t.set_xscale('log')
            ax0t.set_yscale('log') 
            axe.set_xscale('log')
            axe.set_yscale('log')
            axe.set_ylabel('Standardized error of quantiles')
            axe.set_xlabel('P(x)')
            
        axe.get_ylim()

        ax0t.set_xlim(axe.get_xlim())
        top_xticks = axe.get_xticks()
        ax0t.set_xticks(top_xticks)
        ax0t.set_xbound(axe.get_xbound())
        ax0t.set_xticklabels([math.floor(x * train_len) for x in axe.get_xticks()])
        ax0t.set_xlabel('Number of training samples')

    fig.suptitle('Trained with '+str(train_len)+' samples', fontsize=16, y=1.1)
    


""" The function to calculate expected value form the trained model """

def obtain_exp_value(model,cond_state):
    width = 0.1
    xlim = [0.1,30]
    x = np.arange(start=xlim[0], stop=xlim[1], step=width)
    # model pdf
    prob=[]
    for i in range(len(x)):
        mx = np.array([cond_state])
        my = np.array([x[i]])
        prob.append(model.pdf(mx,my))

    return np.sum(x*np.squeeze(prob)*width)

""" Example for exp calculation """
#print(obtain_exp_value(model,[0,1,1]))
#print(obtain_exp_value(gmm_model,[0,1,1]))


def evaluate_models_allstates_agg(cma_results,train_len,n_epoch,xsize=3,quantiles=[0.9, 0.99, 0.999, 0.9999, 0.99999],model_names=['EMM-GPD','GMM','EMM-PL']):
    n_bars = len(cma_results)
    fig, ax = plt.subplots(1, 1, figsize=(8,4))

    # Quantiles bar plot
    labels = [ str(q) for q in quantiles ]
    xbar = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    each_bar_width = width/n_bars
    each_bar_start = xbar - width/2 + each_bar_width/2
    for j in range(n_bars):
        results = cma_results[j]
        model_agg = np.array([statistics.mean(abs(results[:,xsize+i])) for i in range(len(quantiles)) ])
        model_agg_pos = np.array([np.sum(results[:,xsize+i] < 0, axis=0)/len(results[:,xsize+i]) for i in range(len(quantiles)) ])
        model_agg_neg = np.array([np.sum(results[:,xsize+i] >= 0, axis=0)/len(results[:,xsize+i]) for i in range(len(quantiles)) ])
        pbar = ax.bar(each_bar_start + j*each_bar_width, model_agg_pos*model_agg, each_bar_width, label=model_names[j]+">0")
        pbar_color = pbar.patches[0].get_facecolor()
        pbar_color = (pbar_color[0],pbar_color[1],pbar_color[2],0.4)
        ax.bar(each_bar_start + j*each_bar_width, model_agg_neg*model_agg, each_bar_width, label=model_names[j]+"<0", bottom=model_agg_pos*model_agg, color=pbar_color)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Standardized quantiles errors')
    ax.set_xlabel('quantiles')
    ax.set_title('Average Quantile Errors')
    ax.set_xticks(xbar)
    ax.set_xticklabels(labels)
    ax.grid()
    ax.legend()

def evaluate_models_tail_agg_plot_save(tail_results,tail_x_axis,tail_x_quants,model_names=['EMM'],plotylim=[3,0],save_fig_addr=None):

    plt.style.use('plot_style.txt')
    fig, ax = plt.subplots(figsize=(10,5))

    for i in range(len(model_names)):
        ax.plot(1-tail_x_quants, tail_results[i], 'o-', label=model_names[i])
        ax.set_xscale('log')
        ax.invert_xaxis()
        ax.set_xlabel('Quantile [log]')
        ax.set_ylabel('Average tail probability error [log]')
        ax.set_ylim(bottom=plotylim[1], top=plotylim[0])
        ax.set_xlim(left=max(1-tail_x_quants),right=min(1-tail_x_quants))
        ax.grid()
        ax.legend(prop={'size': 20})

    plt.grid()

    if save_fig_addr is not None:
        plt.savefig(save_fig_addr+'tail_agg.png',bbox_inches='tight')
    else:
        plt.show()


def evaluate_models_allstates_plot_save(cma_results,train_len,model_names=['EMM','GMM'],quantiles=[0.9, 0.99, 0.999, 0.9999, 0.99999],xsize=3,markers=['s','o','v','*','x'],loglog=False, ylim_ll=[1e-4,1e0],ylim=[0,0.4],marker_size=80,save_fig_addr=None):

    n_plots = len(cma_results)

    for m in range(n_plots):

        fig, axe = plt.subplots(figsize=(10,5))

        results = cma_results[m]
        # start the plot
        leg_custs = []

        #ax0t = axe.twiny()
        for i in range(len(quantiles)):
            colors_arr = np.where(results[:,xsize+i] > 0, 'green', 'red')
            path = axe.scatter(results[:,-1],abs(results[:,xsize+i]),s=marker_size,c=colors_arr,marker=markers[i])
            # custom legend labels and markers
            
            leg_custs.append(mlines.Line2D([], [], color='black', marker=markers[i], linestyle='None',markersize=6, label=str(quantiles[i])))

        #red_patch = mpatches.Patch(color='red', label='error<0')
        #green_patch = mpatches.Patch(color='green', label='error>0')
        #leg_custs.append(red_patch)
        #leg_custs.append(green_patch)
        #axe.legend(handles=leg_custs)
        axe.legend(handles=leg_custs, ncol=len(leg_custs))
        axe.grid()
        #axe.set_title(model_names[m])

        if loglog is False:
            axe.set_ylim(ylim)
            axe.set_ylabel('Standardized error of quantiles')
            axe.set_xlabel('P(state)')
        else:
            axe.set_ylim(ylim_ll)
            #ax0t.set_xscale('log')
            #ax0t.set_yscale('log') 
            axe.set_xscale('log')
            #xticks = np.logspace(math.log10( min(results[:,-1] ) ), math.log10( max(results[:,-1]) ), num=2)
            #axe.set_xticks(xticks)
            axe.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.3f'))
            #axe.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

            axe.set_yscale('log')
            axe.set_ylabel('Standardized error of quantiles')
            axe.set_xlabel('P(state)')
            
        axe.get_ylim()

        #ax0t.set_xlim(axe.get_xlim())
        #top_xticks = axe.get_xticks()
        #ax0t.set_xticks(top_xticks)
        #ax0t.set_xbound(axe.get_xbound())
        #ax0t.set_xticklabels([math.floor(x * train_len) for x in axe.get_xticks()])
        #ax0t.set_xlabel('Number of training samples')

        #fig.suptitle('Trained with '+str(train_len)+' samples', fontsize=16, y=1.1)

        if save_fig_addr is not None:
            plt.savefig(save_fig_addr+'allstate_'+model_names[m]+'.png',bbox_inches='tight')
        else:
            plt.show()

        


def evaluate_models_allstates_agg_save(cma_results,train_len,n_epoch,xsize=3,quantiles=[0.9, 0.99, 0.999, 0.9999, 0.99999],model_names=['EMM-GPD','GMM','EMM-PL'],ebar_width=0.35,save_fig_addr=None):
    n_bars = len(cma_results)
    fig, ax = plt.subplots(figsize=(10,5))

    # Quantiles bar plot
    labels = [ str(q) for q in quantiles ]
    xbar = np.arange(len(labels))  # the label locations
    width = ebar_width #0.35  # the width of the bars
    each_bar_width = width/n_bars
    each_bar_start = xbar - width/2 + each_bar_width/2
    for j in range(n_bars):
        results = cma_results[j]
        model_agg = np.array([statistics.mean(abs(results[:,xsize+i])) for i in range(len(quantiles)) ])
        model_agg_pos = np.array([np.sum(results[:,xsize+i] < 0, axis=0)/len(results[:,xsize+i]) for i in range(len(quantiles)) ])
        model_agg_neg = np.array([np.sum(results[:,xsize+i] >= 0, axis=0)/len(results[:,xsize+i]) for i in range(len(quantiles)) ])
        #pbar = ax.bar(each_bar_start + j*each_bar_width, model_agg_pos*model_agg, each_bar_width, label=model_names[j]+">0")
        pbar = ax.bar(each_bar_start + j*each_bar_width, model_agg_pos*model_agg, each_bar_width, label=model_names[j])
        pbar_color = pbar.patches[0].get_facecolor()
        pbar_color = (pbar_color[0],pbar_color[1],pbar_color[2],0.4)
        #ax.bar(each_bar_start + j*each_bar_width, model_agg_neg*model_agg, each_bar_width, label=model_names[j]+"<0", bottom=model_agg_pos*model_agg, color=pbar_color)
        ax.bar(each_bar_start + j*each_bar_width, model_agg_neg*model_agg, each_bar_width, bottom=model_agg_pos*model_agg, color=pbar_color)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Standardized quantiles errors')
    ax.set_xlabel('quantiles')
    #ax.set_title('Average Quantile Errors')
    ax.set_xticks(xbar)
    ax.set_xticklabels(labels)
    ax.grid()
    ax.legend(prop={'size': 20})

    if save_fig_addr is not None:
        plt.savefig(save_fig_addr+'quantile_agg.png',bbox_inches='tight')
    else:
        plt.show()
