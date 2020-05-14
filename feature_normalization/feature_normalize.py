import numpy as np
import os


def split_line(line):
    """ Splits a line of a file according to the format "<job_id>\t<feature_type>,<feature_1>,...<feature_n>"
    
    Parameters
    ----------
    line : str
        line to be splitted
        
    Returns
    -------
    item_id : str
        id of the item (job)
    feature_type : str
        type of the item
    features : int[] 
        list of integer feature values
    """
    features = line.split(",")
    item_id, feature_type = features[0].split("\t")
    features = [int(x) for x in features[1:]]
    return item_id, feature_type, features


def _default_output(f, item_id, features, max_feature_index, max_feature_abs_mean_diff):
    row_str = f"{item_id}, {max_feature_index}, {max_feature_abs_mean_diff}, " + \
        ", ".join([str(f) for f in features]) + "\n"
    f.write(row_str)


def _default_header(f, feature_type, features):
    features_header = [f"feature_{feature_type}_stand_{i}" for i in range(len(features))]
    header = f"id_job, max_feature_{feature_type}_index, max_feature_{feature_type}_abs_mean_diff, " +\
                    ", ".join(features_header) + "\n"
    f.write(header)


class FeatureProcessor:
    """ Class that handles feature processing using the given feature normalizer
    """
    def __init__(self, normalizer, split_line=split_line):
        """Parameters
        ----------
        normalizer : Normalizer
            normalizer object that implements required normalization. Should be trained before usage in FeatureProcessor
        split_line : function, optional
            line splitting function with signature item_id, feature_type, features split_line(str). 
            If input file format differs should be overwritten.
        """
        self.norm = normalizer
        self.split_line = split_line
            
    def process(self, file_path, output_file, result_row_handler=None, header_handler=None):
        """ Processes a given input file: normalizes features, 
            extracts max feature index from each row and its difference from the corresponding mean,
            writes to file (by default, can be overwritten)
    
        Parameters
        ----------
        file_path : str
            path to the file to be processed
        output_file : str
            output file path. If file exists, header will not be written, new rows will be appended at the end of the file.
            If file doesn't exists it will be created and header
        result_row_handler : function, optional
            function that handles processed line of the original file, e.g. writes it somewhere.
            Required signature is output_handler(file, str, str, float[], int, float).
            Default handler writes all rows to the specified file
        header_handler : function, optional
            function that handles result file header creation and writing.
            Required signature is header_handler(file, str, float[]).
            Default handler writes header of the following format:
            "id_job, max_feature_{feature_type}_index, max_feature_{feature_type}_abs_mean_diff,
                ...feature_{feature_type}_stand_{i}..."
        
        """
        new_file = not os.path.isfile(output_file)
        with open(file_path, "r") as f, open(output_file, "a") as f_out:
            next(f)

            for line in f:
                item_id, feature_type, features, max_feature_index, max_feature_abs_mean_diff = self._process_line(line)
                if new_file:
                    if header_handler is None:
                        _default_header(f_out, feature_type, features)
                    else:
                        header_handler(f_out, feature_type, features)
                    new_file = False
                    
                if result_row_handler is None:
                    _default_output(f_out, item_id, features, max_feature_index, max_feature_abs_mean_diff)
                else:
                    result_row_handler(f_out, item_id, feature_type, features,
                                       max_feature_index, max_feature_abs_mean_diff)

    def _process_line(self, line):
        item_id, feature_type, features = self.split_line(line)
        max_feature_index = np.argmax(features)
        max_feature_abs_mean_diff = self.norm.mean_diff(feature_type, max_feature_index, features[max_feature_index])
        
        for i in range(len(features)):
            features[i] = self.norm.normalize(feature_type, i, features[i])
        
        return item_id, feature_type, features, max_feature_index, max_feature_abs_mean_diff


class Normalizer:
    """ Class that defines high-level interface for feature normalizer.
    Provides default implementation for learning feature means from file and calculation of the difference between mean and given value.
    """
    def __init__(self, split_line=split_line):
        """Parameters
        ----------
        split_line : function, optional
            line splitting function with signature item_id, feature_type, features split_line(str). 
            If input file format differs should be overwritten.
        """
        self.means = {}
        self.split_line = split_line
        
    def fit_mean(self, train_file_path):
        """ Learns features means
    
        Parameters
        ----------
        train_file_path : str
            path to the file to be processed
        """
        with open(train_file_path, "r") as f:
            next(f)
            counter = 0
            sums = None
            feature_type = None

            for line in f:
                item_id, feature_type, features = self.split_line(line)
                if sums is None:
                    sums = np.zeros_like(features)
                sums += features
                counter += 1
            
            self.means[feature_type] = sums / counter
            
    def normalize(self, feature_type, feature_index, feature_value):
        pass
    
    def mean_diff(self, feature_type, feature_index, feature_value):
        """ Calculates the differnce between the given feature value and the corresponding mean
        """
        means = self.means.get(feature_type)
        if means is None or feature_index >= len(means):
            raise Exception(f"No mean value available for feature #{feature_index} of type {feature_type}. Please, fit normalizer properly before using it")
        else:
            return abs(means[feature_index] - feature_value)


class ZNormalizer(Normalizer):
    """Implementation of z-norma
    """
    def __init__(self):
        self.stds = {}
        super(ZNormalizer, self).__init__()
        
    def fit(self, train_file_path):
        self.fit_mean(train_file_path)
        self.fit_std(train_file_path)

    def fit_std(self, train_file_path):
        """ Learns features std.
        Standard deviation is corrected sample standard deviation of train data.
        I.e. sum of squared differences is divided by (n-1), not n.
        This provides unbiased estimate for std, especially important for small train data.
    
        Parameters
        ----------
        train_file_path : str
            path to the file to be processed
        """
        with open(train_file_path, "r") as f:
            next(f)
            counter = 0
            sums = None
            feature_type = None
            
            for line in f:
                item_id, feature_type, features = self.split_line(line)
                if sums is None:
                    sums = np.zeros_like(features, dtype='float64')
                sums += np.power((features - self.means[feature_type]), 2)
                counter += 1

            denom = (counter-1) if counter > 1 else 1
            self.stds[feature_type] = np.sqrt(sums / denom)
        
    def normalize(self, feature_type, feature_index, feature_value):
        """ Calculates z-normalized value for the given feature using the corresponding mean and std
        """
        means = self.means.get(feature_type)
        stds = self.stds.get(feature_type)
        if means is None or feature_index >= len(means) or stds is None or feature_index >= len(stds):
            raise Exception(f"No mean or std value available for feature #{feature_index} of type {feature_type}. "
                            f"Please, fit normalizer properly before using it")
        else:
            return (feature_value - self.means.get(feature_type)[feature_index]) \
                   / self.stds.get(feature_type)[feature_index]
