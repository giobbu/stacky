from loguru import logger

class BasicHashing:
    " A class to implement basic hashing functions "
    def __init__(self, n_buckets):
        self.n_buckets = n_buckets

    def hash_function(self, value):
        hash_value =  int(value) % self.n_buckets
        logger.info(f"Hash value for {value} is {hash_value}")
        return hash_value

    def basic_hash_table(self, value_l):
        # Initialize all the buckets in the hash table as empty lists
        hash_table = {i:[] for i in range(self.n_buckets)} 
        for value in value_l:
            hash_value = self.hash_function(value) 
            hash_table[hash_value].append(value)
        return hash_table
    
if __name__=='__main__':
    # Create a basic hash table with 5 buckets
    n_buckets = 5
    basic_hash = BasicHashing(n_buckets)
    # input values to hash
    value_l = [100, 10, 14, 17, 97, 23, 27]
    # create a basic hash table
    hash_table = basic_hash.basic_hash_table(value_l)
    # print the hash table
    print(hash_table)