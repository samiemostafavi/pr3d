import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


def create_dataset(n_samples = 300, x_dim=3, x_max = 10, x_level=2, dtype = 'float64', dist = 'normal'):

    # generate random sample, two components
    X = np.array(np.random.randint(x_max, size=(n_samples, x_dim))*x_level).astype(dtype)

    if dist is 'normal':
        Y = np.array([ 
                np.random.normal(loc=x_sample[0]+x_sample[1]+x_sample[2],scale=(x_sample[0]+x_sample[1]+x_sample[2])/5)
                    for x_sample in X 
            ]
        ).astype(dtype)
    elif dist is 'gamma':
        Y = np.array([ 
                np.random.gamma(shape=x_sample[0]+x_sample[1]+x_sample[2],scale=(x_sample[0]+x_sample[1]+x_sample[2])/5)
                    for x_sample in X 
            ]
        ).astype(dtype)

    return X,Y


""" load parquet dataset """
def load_parquet(file_addresses, read_columns=None):

    table = pa.concat_tables(
        pq.read_table(
            file_address,columns=read_columns,
        ) for file_address in file_addresses
    )

    return table.to_pandas()
    
