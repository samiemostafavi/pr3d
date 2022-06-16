import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import matplotlib.pyplot as plt
import seaborn as sns

from pr3d.nonbayesian import ConditionalGammaEVM

# load dataset first
file_addresses = ['dataset_onehop_processed.parquet']
table = pa.concat_tables(
    pq.read_table(
        file_address,columns=None,
    ) for file_address in file_addresses
)
df = table.to_pandas()
print(df)

# load the trained model
dtype = 'float64'
conditional_delay_model = ConditionalGammaEVM(
    h5_addr = "onehop_tis_model.h5",
)

# find n most common queue_length occurances
n = 3
values_count = df[['queue_length']].value_counts()[:n].index.tolist()
print("{0} most common queue states: {1}".format(n,values_count))

# divide the service delay into n segments based on quantiles
m = 5
service_delays = np.squeeze(df[['service_delay']].to_numpy())
quants = np.linspace(0, 1, num=m+1)
intervals = [ (quant,quants[idx+1]) for idx, quant in enumerate(quants) if (idx+1)<len(quants) ]
print("{0} longer_delay_prob intervals: {1}".format(n,intervals))

#sns.set_palette("rocket")

# plot the conditional distributions of them
fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(m*4,n*4))

for i in range(n):
    for j in range(m):
        ax = axes[i,j]

        # take the empirical samples
        conditional_df = df[
            (df.queue_length==values_count[i][0]) & 
            (df.longer_delay_prob>=intervals[j][0]) & 
            (df.longer_delay_prob<intervals[j][1])
        ]

        # sample the predictor with x (conditions) from the empirical data
        X = np.squeeze(conditional_df[['queue_length','longer_delay_prob']].to_numpy())
        conditional_samples = conditional_delay_model.sample_n(
            x = X,
            random_generator=np.random.default_rng(0),
        )

        # insert it to the dataset
        conditional_df['predicted distribution'] = conditional_samples

        conditional_df.rename(columns = {'end2end_delay':'empirical distribution'}, inplace = True)

        # plot
        sns.histplot(
            conditional_df[['empirical distribution','predicted distribution']],
            kde=True, 
            ax=ax,
            stat="density",
        ).set(title="x={}, interval={}, count={}".format(
            values_count[i],
            ["{:0.2f}".format(inter) for inter in intervals[j]],
            len(conditional_df))
        )
        ax.title.set_size(10)


fig.tight_layout()
plt.savefig('conditional_delay_tis.png')