from tokenize import Double
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
import pyarrow as pa

""" load dataset """
class ParquetDataset():
    def __init__(self, file_addresses, predictor_num = 1):
        for file_address in file_addresses:
            if predictor_num is 1:
                table = pa.concat_tables(
                    pq.read_table(file_address,columns=['end2enddelay','h1_uplink_netstate','h1_compute_netstate','h1_downlink_netstate'])
                    for file_address in file_addresses)
            elif predictor_num is 2:
                table = pa.concat_tables(
                    pq.read_table(file_address,columns=['totaldelay_compute','totaldelay_downlink','h2_compute_netstate','h2_downlink_netstate'])
                    for file_address in file_addresses)
                table = table.add_column(0,'delay', pc.add(table.column('totaldelay_compute'),table.column('totaldelay_downlink'))).drop(['totaldelay_compute','totaldelay_downlink'])
            elif predictor_num is 3:
                table = pa.concat_tables(
                    pq.read_table(file_address,columns=['totaldelay_downlink','h3_downlink_netstate'])
                    for file_address in file_addresses)
                
        self.dataset = table.to_pandas().to_numpy()

        self.predictor_num = predictor_num
        self.n_records = len(self.dataset)
        self.n_features = len(self.dataset[0])
        #print('Predictor-%d dataset loaded from .parquet file. Rows: %d ' %(predictor_num,len(self.dataset)), ' Columns: %d ' % len(self.dataset[0]))

    def get_data(self, n_samples, n_replicas):
        """
        This function returns a vector of n_samples from the dataset randomly
        """
        result = np.empty((n_samples,self.n_features,n_replicas))
        for i in range(n_replicas):
            idx = np.random.randint(self.n_records, size=n_samples)
            result[:,:,i] = self.dataset[idx,:]
        return result

    def get_data_leg(self, n_samples):
        """
        This function returns a vector of n_samples from the dataset randomly
        """
        idx = np.random.randint(self.n_records, size=n_samples)
        return self.dataset[idx,:]
    def get_data_servicerate_cond(self, s_cond):
        """
        This function returns a vector of n_samples from the dataset randomly
        """

        idx = []
        new_ds = self.dataset[:,[1,2,3]]
        for i in range(len(new_ds)):
            if all(new_ds[i] == s_cond):
                idx.append(i)
        cond_res = self.dataset[idx,:]

        idz = np.random.randint(len(cond_res), size=len(cond_res))
        return cond_res[idz,:]

    def get_data_firstrows_cond(self, n_samples):
        """
        This function returns a vector of n_samples from the dataset randomly
        """

        new_ds = self.dataset[0:n_samples,:]
        idx = np.random.randint(len(new_ds), size=len(new_ds))
        return new_ds[idx,:]

    def get_data_unshuffled(self, n_samples):
        """
        This function returns the first n_samples from the dataset 
        """
        return self.dataset[0:n_samples,:]