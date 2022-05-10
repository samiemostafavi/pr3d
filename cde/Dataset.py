import numpy as np

def create_dataset(n_samples = 300, x_dim=3, x_max = 10, x_level=2):

    # generate random sample, two components
    X = np.array(np.random.randint(x_max, size=(n_samples, x_dim))*x_level)

    Y = np.array([ 
            np.random.normal(loc=x_sample[0]+x_sample[1]+x_sample[2],scale=(x_sample[0]+x_sample[1]+x_sample[2])/5) 
                for x_sample in X 
        ]
    )

    return X,Y