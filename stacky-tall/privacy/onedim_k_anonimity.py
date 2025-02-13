import numpy as np
import pandas as pd

class OneDKAnonymity:
    " One-Dimensional K-Anonymity Algorithm for Privacy Preserving Data Publishing with Mondrian Partitioning and continuous quasi-identifiers"
    def __init__(self, data: pd.DataFrame, k: int, qi: str):
        assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
        self.data = data
        self.k = k
        self.qi = qi

    def mondrian_partition(self, data: pd.DataFrame):
        partitions = []
        # If the data is less than or equal to k, return the data
        if len(data) <= (2 * self.k-1):
            partitions.append(data)
            return [data]
        # Sort the data
        data = data.sort_values(by=self.qi)
        # Get the total count
        total_count = data[self.qi].count()
        # Find the mid-point
        mid = total_count // 2
        # Split the data
        data_left = data[:mid]
        data_right = data[mid:]
        # Recursively partition the data
        partitions.extend(self.mondrian_partition(data_left))
        partitions.extend(self.mondrian_partition(data_right))
        return partitions
    
    def anonymize(self):
        partitions = self.mondrian_partition(self.data)
        list_anonymized_data = []
        for i, partition in enumerate(partitions):
            min_value = partition[self.qi].min()
            max_value = partition[self.qi].max()
            range_value = f"[{min_value} - {max_value}]" if min_value != max_value else f"[{min_value}]"
            partitions[i][self.qi] = range_value
            list_anonymized_data.append(partition)
        # Concatenate the partitions to one DataFrame
        anonymized_data = pd.concat(list_anonymized_data, ignore_index=True)
        return anonymized_data
    
if __name__== "__main__":
    # create data for testing
    import numpy as np
    import pandas as pd
    col1 = np.random.randint(20, 60, 100)
    col2 = np.random.randint(1, 100, 100)
    data = pd.DataFrame({'age': col1, 'income': col2})
    k = 10
    qi = "age"
    one_d_k_anonymity = OneDKAnonymity(data, k, qi)
    anonymized_data = one_d_k_anonymity.anonymize()
    print(data.sort_values(by='income'))
    print('--'*20)
    print(anonymized_data.sort_values(by='income'))


