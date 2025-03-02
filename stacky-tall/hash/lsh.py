import numpy as np
from loguru import logger

class LocalitySensitiveHashing:
    """ Locality Sensitive Hashing Algorithm """

    def __init__(self, seed, num_hash_tables, num_hyperplanes, num_dimensions):
        self.seed = seed
        self.num_hash_tables = num_hash_tables
        self.num_hyperplanes = num_hyperplanes
        self.num_dimensions = num_dimensions
        logger.info(f"LSH with {num_hash_tables} hash tables, {num_hyperplanes} hyperplanes, and {num_dimensions} dimensions.")

    def _generate_hyperplanes(self):
        " Generate hyperplanes "
        np.random.seed(self.seed)
        universe_planes = [np.random.normal(size=(self.num_dimensions, self.num_hyperplanes)) for _ in range(self.num_hash_tables)]
        return universe_planes

    def _generate_hash_value(self, vec, planes):
        "Create a hash for a vector; hash_id says which random hash to use."
        # for each set of planes, calculate the sign of the dot product between the vector and the planes
        dot_product = np.dot(vec, planes)
        sign_of_dot_product = np.sign(dot_product)
        # set h to be false (eqivalent to 0 when used in operations) if the sign is negative,
        # and true (equivalent to 1) if the sign is positive
        # if the sign is 0, i.e. the vector is in the plane, consider the sign to be positive
        h = sign_of_dot_product >=0
        # remove extra un-used dimensions (convert this from a 2D to a 1D array)
        h = np.squeeze(h)
        # initialize the hash value to 0
        hash_value = 0
        n_planes = len(h)
        for i in range(n_planes):
            # increment the hash value by 2^i * h_i        
            hash_value += 2**i *h[i]
        # cast hash_value as an integer
        hash_value = int(hash_value)
        return hash_value
    
    def _create_hash_table(self, vecs, planes):
        " Create a hash table for the input vectors"
        # number of planes is the number of columns in the planes matrix
        num_of_planes = planes.shape[1]
        # number of buckets is 2^(number of planes)    
        num_buckets = 2**num_of_planes
        # create the hash table as a dictionary.
        hash_table = {key: [] for key in range(num_buckets)}
        # create the id table as a dictionary.
        id_table = {key: [] for key in range(num_buckets)}
        # for each vector in 'vecs'
        for i, v in enumerate(vecs):
            # calculate the hash value for the vector
            h = self._generate_hash_value(v, planes)
            # store the vector into hash_table at key h,
            hash_table[h].append(v)
            # store the vector's index 'i' (each document is given a unique integer 0,1,2...)
            id_table[h].append(i)
        return hash_table, id_table

    def create_hash_tables(self, vecs):
        " Create multiple hash tables for the input vectors"
        hash_tables = []
        id_tables = []
        planes_l = self._generate_hyperplanes()
        for table_id in range(self.num_hash_tables):
            logger.info(f"Working on hash table #: {table_id}")
            planes = planes_l[table_id]
            hash_table, id_table = self._create_hash_table(vecs, planes)
            hash_tables.append(hash_table)
            id_tables.append(id_table)
        return hash_tables, id_tables
    

if __name__=='__main__':
    # vector embeddings
    vecs = np.random.normal(0, 1, (1000, 100))
    logger.info(f"Data shape: {vecs.shape}")
    logger.info(f"Example vector: {vecs[0]}")
    # number of hash tables
    num_hash_tables = 5
    # number of hyperplanes
    num_hyperplanes = 10
    # number of dimensions
    num_dimensions = 100
    # seed
    seed = 42
    # initialize LSH class
    lsh = LocalitySensitiveHashing(seed, num_hash_tables, num_hyperplanes, num_dimensions)
    # create hash tables
    hash_tables, id_tables = lsh.create_hash_tables(vecs)
    logger.info(f"Hash tables: {id_tables[0]}")

