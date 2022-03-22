from typing import Set, Tuple, List, Union, Dict
from itertools import combinations

import skbio
import pandas as pd
import numpy as np

from tqdm import tqdm
from probables import ExpandingBloomFilter

import yaml

def scalar_constructor(loader, node):                                           
    value = loader.construct_scalar(node)                                       
    return value                                                                
                                                                                
yaml.add_constructor('!ref', scalar_constructor)                                
yaml.add_constructor('!no-provenance', scalar_constructor)                      
yaml.add_constructor('!color', scalar_constructor)                              
yaml.add_constructor('!cite', scalar_constructor)                               
yaml.add_constructor('!metadata', scalar_constructor)



class AnankeBloomFilter(object):
    """This class is a wrapper for pyprobables ExpandingBloomFilter
    adding additional functionality for tracking pairs at a threshold.

    An AnankeBloomFilter (ABF) is a class that stores the pairing
    information for a large set of objects across a series of bloom
    filters for various thresholds of an arbitrary distance measure

    Args:
        threshold (float): A range of distance thresholds to sample the pairwise relationships
        false_positive_rate (float): The false positive rate for pairs in the worst case; determines when a bloom filter is full and is expanded
        manager (AnankeManager): The AnankeManager class object that organizes the set of filters this bloom belongs to
        bloom (ExpandingBloomFilter, optional): An ExpandingBloomFilter, supplied if loading from file. If None (default), will make a new, empty ExpandingBloomFilter.
    """

    
#    Attributes:
#        threshold (float): A range of distance thresholds to sample the pairwise relationships, in ascending order
#        false_positive_rate (float): The false positive rate for pairs in the worst case; determines when a bloom filter is full and is expanded
#        bloom (ExpandingBloomFilter): A list of bloom filters, the same length and order as self.thresholds, representing the object neighbourhoods at those thresholds
#        manager (AnankeManager): The AnankeManager that manages this bloom filter

    __slots__ = ["_fpr","_blm","_thrshld","_mgr"]
    def __init__(self, 
                 threshold: float, 
                 manager,#: AnankeManager, #Type hint here causes circular reference
                 bloom: Union[None, ExpandingBloomFilter] = None, 
                 false_positive_rate: float = 0.01):
        self._fpr = false_positive_rate
        self._thrshld = threshold
        self._mgr = manager
        #If no objects, then set it to handle 10,000 pairs as a minimum
        #If objects, set it to 1/3 the number of unique pairs (sparsity of 0.5)
        est_elements = max(1e4, 
                           (len(self.manager._objs)*(len(self.manager._objs)-1))/4)
        if bloom == None:
            self._blm = ExpandingBloomFilter(est_elements=int(est_elements), 
                                             false_positive_rate=self._fpr)
        else:
            self._blm = bloom
            
    @property
    def false_positive_rate(self) -> float:
        """ float: The desired false positive rate of the expanding Bloom Filter """
        return self._fpr
    
    @property
    def threshold(self) -> float:
        """ float: The distance threshold by which a pair is determined to be 
                   neighbours in this filter """
        return self._thrshld
    
    @property
    def bloom(self) -> ExpandingBloomFilter:
        """ ExpandingBloomFilter: The bloom filter that stores pair information"""
        return self._blm
    
    @property
    def manager(self): # -> AnankeManager: Type hint here causes circular reference
        """ AnankeManager: The AnankeManager object that manages this bloom filter"""
        return self._mgr
    
    def info(self) -> str:
        """Return an information string detailing the size of the AnankeBloomFilter
           object.

        Returns:
            A string containing detailed info for the AnankeBloomFilter object.

        """
        n_added = self._blm.elements_added
        info_str = "Bloom filter for distance measure '%s', threshold %f, " \
                   "storing %d pairs, " \
                   "with capacity for %d before expansion is required. " \
                   "Using %.2f MB of memory." % (self.manager.distance_measure,
                                                  self.threshold,
                                                  n_added,
                                                  self._blm.estimated_elements,
                                                  self.memsize(in_bytes=True)/(1024**2))
        return info_str
    
    def are_neighbours(self, objA: str, objB: str) -> bool:
        """Returns True if `objA` and `objB` are neighbours, else False.
           Pair relationships are always symmetrical.

        Args:
            objA: The name of the first object.
            objB: The name of the second object.

        Returns:
            True if objA and objB are neighbours at self.threshold, otherwise False.

        """
        return "__".join(np.sort([objA,objB])) in self._blm
    
    def are_neighbors(self, objA: str, objB: str) -> bool:
        return self.are_neighbours(*args)
    
    def all_neighbours(self, obj: str) -> bool:
        neighbours = []
        for objB in self.objects:
            if self.are_neighbours(obj, objB):
                neighbours.append(objB)
        return neighbours
    
    def all_neighbors(self):
        return self.all_neighbours(*args, **kwargs)
    
    def add_neighbours(self, objA: str, objB: str) -> None:
        # Get object names in a consistent order with delimiter
        # This is the object that is hashed and then inserted
        # to represent the pairs
        hashable = "__".join(np.sort([objA,objB]))
        self._blm.add(hashable, force=True)
    
    def add_neighbors(self, objA: str, objB: str) -> None:
        self.add_neighbours(*args)
        
    def _add_alt(self, hashes):
        self._blm.add_alt(hashes, force=True)
    
    def add_neighbourhood(self, list_of_pairs: List[ Tuple[str] ]) -> None:
        for pair in list_of_pairs:
            self.add_neighbours( pair[0], pair[1] )
    
    def add_neighborhood(self) -> None:
        return self.add_neighbourhood(*args)
    
    def memsize(self, in_bytes: bool=True) -> int:
        bloom_total = 0
        for bloom in self._blm._blooms:
            bloom_total += len(bloom._bloom)
        if in_bytes:
            return bloom_total / 8
        return bloom_total
    
    def to_binary_file(self, out_file):
        self._blm.export(out_file)

class Ananke(object):
    """Ananke is the primary class of Ananke.

    An Ananke object controls the querying of a set of bloom filters
    that record the pairwise relationships for a given distance
    measure and set of objects (e.g., samples or features).
    
    Args:
        object_type (str): 'samples' or 'features', required.
        distance_measure (str): The distance measure that is used 
            to compute the pairs, required unless distance_matrix
            or tree are provided.
        thresholds (:obj:`list` of :obj:`float`): A range of
            distance thresholds to sample the pairwise
            relationships
        threshold_method (str): The method to use to select the
            distance thresholds to sample: [min_max, mean_var, mean_stddev]
        n_filters (int): The number of filters, evenly spaced, if
            thresholds is not provided
        variance_multiplier (float): A multiplier for the 'mean_var'
            and 'mean_stddev' threshold_method [default: 1.0]
        false_positive_rate (float): The false positive rate for
            pairs in the worst case; determines when a bloom 
            filter is full and is expanded. This is the best way
            to control bloom filter size in-memory (smaller false
            positive rate means more memory used).
        objects (:obj:`list` of :obj:`str`, optional): A list of 
            the initial samples or features whose pairwise 
            relationships are being stored
        biom_table (:obj: biom.Table): A BIOM formatted table
    """
#    Attributes:
#        distance_measure (str): The distance measure that is used 
#            to compute the pairs
#        thresholds (:obj:`list` of :obj:`float`): A range of
#            distance thresholds to sample the pairwise
#            relationships, in ascending order
#        false_positive_rate (float): The false positive rate for
#            pairs in the worst case; determines when a bloom 
#            filter is full and is expanded
#        objects (:obj:`set` of :obj:`str`): A set of the initial
#            samples or features whose pairwise relationships are 
#            being stored
#        blooms (:obj:`list` of :obj:`ExpandingBloomFilter`): A list
#            of bloom filters, the same length and order as 
#            self.thresholds, representing the object neighbourhoods
#            at those thresholds
        
    
    DEFAULT_ESTIMATED_ELEMENTS = 1e4 #By default make room to store 10,000 pairs
    __slots__ = ["_dm", "_fpr", "_objs", "_blms", "_pair_dist",
                 "_thrshlds", "_obj_type", "_table", "_tree", "_tree_dm",
                 "_src"]
    def __init__(self, object_type = None,
                       thresholds = None,
                       false_positive_rate = 0.01,
                       distance_measure = None,
                       threshold_method = 'mean_stddev',
                       variance_multiplier = 1.0,
                       n_filters = 10,
                       biom_table = None, 
                       distance_matrix = None,
                       tree = None,
                       metadata = None,
                       ananke_artifact = None):
        
        ######### IMPORT CONTROL #########
        
        ##### Ananke Artifact (full save restore) #####
        if ananke_artifact != None:
            # Other filetypes must be omitted
            assert (biom_table == None) and (distance_matrix == None) and (tree == None)
            print("Initializing Ananke object from previously saved Ananke artifact")
            self._from_artifact(ananke_artifact)
            
        ##### BIOM table artifact #####
        elif biom_table != None:
            assert (ananke_artifact == None) and (distance_matrix == None) and (tree == None)
            assert object_type in ["features","samples"]
            self._obj_type = object_type
            # Sets self._objs
            print("Initializing Ananke object from BIOM table artifact")
            self._preprocess_biom_table(biom_table)
        
        ##### Tree artifact #####
        elif tree != None:
            assert (ananke_artifact == None) and (distance_matrix == None) and (biom_table == None)
            # Providing a tree supposes the objects are features
            assert object_type == "features"
            self._obj_type = object_type
            print("Initializing Ananke object from tree artifact")
            self._preprocess_tree(tree)
            
        ##### Distance matrix artifact #####
        elif distance_matrix != None:
            assert (ananke_artifact == None) and (biom_table == None) and (tree == None)
            assert object_type == "samples"
            self._obj_type = object_type
            print("Initializing Ananke object from distance matrix artifact")
            self._preprocess_distance_matrix(distance_matrix)
            
        ##### Empty initialize #####
        else:
            #Empty object, no source file
            assert object_type in ["features","samples"]
            self._obj_type = object_type
            self._objs = set()
            print("Initializing empty Ananke object")
            self._src = "none"
            
        # If sampling thresholds are not given, infer good guesses using the distances
        if (thresholds == None) or (len(thresholds)==0):
            self._thrshlds = self._default_thresholds(threshold_method, n_filters, variance_multiplier)
        else:
            self._thrshlds = thresholds
        
        # Store blooms in a dict
        self._blms = {}
        
        for threshold in self._thrshlds:
            bloom = AnankeBloomFilter(threshold = threshold, 
                                      false_positive_rate = false_positive_rate,
                                      manager = self)
            self._blms[threshold] = bloom
        
    @property
    def object_type(self) -> str:
        """ str: The type of object being tracked by this filter: [samples, features] """
        return self._obj_type
    
    @property
    def objects(self) -> Set [ str ]:
        """ (:obj:`set` of str): The names of all objects tracked by these filters """
        return self._objs
    
    @property
    def distance_measure(self) -> str:
        """ str: The distance measure this manager is tracking"""
        return self._dm
    
    @property
    def thresholds(self) -> List [ float ]:
        """ (:obj:`list` of :obj:`float`): The distance thresholds that indexed
                                           by this Ananke object"""
        return self._thrshlds
    
    @property
    def blooms(self) -> Dict [ float, AnankeBloomFilter ]:
        """ (:obj:`dict` of :obj:`AnankeBloomFilter`): Dict with thresholds as keys to AnankeBloomFilters"""
        return self._blms
    
    def _default_thresholds(self, method, n_filters, variance_multiplier):
        def welford_update(existingAggregate, newValue):
            (count, mean, M2, min_val, max_val) = existingAggregate
            count += 1
            delta = newValue - mean
            mean += delta / count
            delta2 = newValue - mean
            M2 += delta * delta2
            if newValue < min_val:
                min_val = newValue
            if newValue > max_val:
                max_val = newValue
            return (count, mean, M2, min_val, max_val)

        # Retrieve the mean, variance and sample variance from an aggregate
        def welford_finalize(existingAggregate):
            (count, mean, M2, min_val, max_val) = existingAggregate
            if count < 2:
                return float("nan")
            else:
                (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
                return (mean, variance, sampleVariance, min_val, max_val)
        
        self._load_distances()
            
        running = (0,0,0,1e10,0)
        print("Computing mean and variance with Welford's algorithm")
        unique_pairs = combinations(self._objs,2)
        n_unique_pairs = len(self._objs)
        n_unique_pairs = (n_unique_pairs*(n_unique_pairs-1))/2
        with tqdm(total=n_unique_pairs) as t:
            for p in unique_pairs:
                dist=self._pair_dist(p[0],p[1])
                running=welford_update(running, dist)
                t.update()

        mean, variance, sampleVariance, min_val, max_val = welford_finalize(running)
        if method == 'min_max':
            return np.linspace(min_val, max_val, n_filters)
        elif method == 'mean_var':
            return np.linspace(max(min_val, mean-variance_multiplier*variance), 
                               min(max_val, mean+variance_multiplier*variance), 
                               n_filters)
        elif method == 'mean_stddev':
            return np.linspace(max(min_val, mean-variance_multiplier*np.sqrt(variance)), 
                               min(max_val, mean+variance_multiplier*np.sqrt(variance)),
                               n_filters)
        
    def _load_distances(self):
        if self._src == 'tree':
            print("Loading a tree's distance matrix into memory")
            dm = self._tree.tip_tip_distances()
            print("Using %.2f MB on tree distance matrix floats" % (dm.data.size/(1024*1024)))
            self._tree_dm = dm
            pair_dist = lambda x,y: self._tree_dm[x,y]
            self._pair_dist = pair_dist
            
    def _unload_distances(self):
        if self._src == 'tree':
            print("Removing tree distance matrix from memory")
            del self._pair_dist
            del self._tree_dm
    
    def update(self):
        print(self._objs)
        
    def to_artifact(self, ananke_artifact):
        pass
    
    def _from_artifact(self, ananke_artifact):
        #Note: not intended to be called by user. 
        #      Initiate this helper function
        #      by providing an ananke_artifact to __init__
        raise NotImplemented
    
    def _preprocess_biom_table(self, biom_table) -> None:
        # Sets the object list from the biom_table object
        # and stores the table as an attribute
        from qiime2 import Artifact
        biom_table = Artifact.load(biom_table).view(pd.DataFrame)
        if self.object_type == "features":
            objects = biom_table.index.tolist()
        elif self.object_type == "samples":
            objects = biom_table.columns.tolist()
        self._objs = set(objects)
        self._table = biom_table
        self._src = "biom"
        
    def add_biom_table(self, biom_table):
        pass
    
    def add_distance_matrix(self, distance_matrix):
        pass
    
    def _preprocess_distance_matrix(self, distance_matrix):
        def update_params(yaml_obj):
            params = {}
            for d in yaml_obj['action']['parameters']:
                params.update(d)
            return params
        from qiime2 import Artifact
        dm = Artifact.load(distance_matrix)
        
        # Scrape YAML files to attempt to infer/find the distance metric
        yaml_action = yaml.load(open(str(dm._archiver.provenance_dir)+
                                     "/action/action.yaml"), Loader=yaml.Loader)
        params = update_params(yaml_action)
        if 'metric' in params:
            self._dm = params['metric']
        else:
            alias_uuid = yaml_action['action']['alias-of']
            yaml_alias = yaml.load(open(str(dm._archiver.provenance_dir)+
                                        "/artifacts/"+alias_uuid+"/action/action.yaml"), Loader=yaml.Loader)
            params = update_params(yaml_alias)
            if 'metric' in params:
                self._dm = params['metric']
            else:
                alias_uuid = yaml_alias['action']['alias-of']
                yaml_alias = yaml.load(open(str(dm._archiver.provenance_dir)+"/artifacts/"+
                                            alias_uuid+"/action/action.yaml"), Loader=yaml.Loader)
                params = update_params(yaml_alias)
                if 'metric' in params:
                    self._dm = params['metric']
                else:
                    self._dm = 'unknown' #Should we scrape further up? This is typical of pipeline distance matrices
        dm_df = dm.view(skbio.DistanceMatrix).to_data_frame()
        self._objs = set(dm_df.index.tolist())
        self._src = "distance_matrix"
        
    def _update_tree(self):
        if self._src != "tree":
            return
        if not hasattr(self, "_tree_dm"):
            self._load_distances()
        unique_pairs = combinations(self._objs,2)
        n_unique_pairs = len(self._objs)
        n_unique_pairs = (n_unique_pairs*(n_unique_pairs-1))/2
        with tqdm(total=n_unique_pairs) as t:
            for p in unique_pairs:
                dist=self._pair_dist(p[0],p[1])
                self.add_to_blooms(p[0],p[1],dist)
                t.update()
        
    def add_to_blooms(self, objA, objB, dist):
        hashable = "__".join(np.sort([objA,objB]))
        hashes = None
        for bloom_cutoff in self._blms.keys():
            if dist <= bloom_cutoff:
                if hashes == None:
                    hashes = list(self._blms.values())[0]._blm._blooms[0].hashes(hashable)
                self._blms[bloom_cutoff]._add_alt(hashes)
    
    def _preprocess_tree(self, tree):
        from qiime2 import Artifact
        tree = Artifact.load(tree).view(skbio.TreeNode)
        feature_names = [x.name for x in tree.tips()]
        self._dm = "phylogenetic"
        self._objs = set(feature_names)
        self._tree = tree
        self._src = "tree"
        
    def info(self):
        pass
    
    def neighbours(self):
        pass
    
    def neighbors(self, **kwargs):
        #For the Americans
        return self.neighbours(*args, **kwargs)

