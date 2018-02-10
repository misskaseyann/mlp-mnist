import re
import os
import numpy as np

class Read(object):
    """
    Basic utility of reading in txt/csv files of data.
    This will produce a label array and normalized data array for use in ML.
    Specifically, this class is modeled after the MNIST data.
    """
    def __init__(self, columns, rows, numpixels, path_label, path_data):
        """
        Initialize all variables for reading the incoming
        data file.

        :param columns: number of columns/features.
        :param rows: number of rows/inputs.
        :param numpixels: the total range of pixels. If using
        unmodified data set from MNIST, this value would be 255.
        :param path_label: file path to labels being read.
        :param path_data: file path to inputs being read.
        """
        self.columns = columns
        self.rows = rows
        self.numpixels = numpixels
        #  Mac/Unix users will change this: /Users/user_name/Documents/file
        #  Windows and Linux: I have no idea and you are out of luck!
        self.path_label = "/Users/kaseystowell/Documents/workspace/mlp-mnist/dataset/" + path_label
        self.path_data = "/Users/kaseystowell/Documents/workspace/mlp-mnist/dataset/" + path_data
        #  Storage of the raw io data.
        self.label_raw_data = None
        self.data_raw_data = None
        #  Target data is kept here.
        self.targetarr = np.empty([self.rows, 10])
        #self.targetarr = [[0] * 1 for i in range(rows)]
        #  Data for each feature is kept here.
        self.inputarr = np.empty([self.rows, self.columns])

    def run(self):
        """
        Runs the Read program in full:
        1. Read the data from the filepaths.
        2. Split the strings in accordance to the data and store in arrays.
        3. Normalize all input values.

        :return: target array and input array.
        """
        self.label_raw_data = self.read_data(self.path_label)
        self.data_raw_data = self.read_data(self.path_data)
        if self.label_raw_data == None or self.data_raw_data == None:
            print("Cannot run program without data.")
        else:
            self.split_string(30000)
            self.normalize()
            return self.targetarr, self.inputarr

    def read_data(self, path):
        """
        Read the file path for a txt file.

        :param p: path to txt file.
        :return: readable file stream.
        """
        if os.path.isfile(path):
            file = open(path, 'r+', encoding='utf-8')
            raw = file.read()
            return raw

        else:
            print("This is not a valid file path or file.")

    def split_string(self, target):
        """
        Split up the string from a read txt file and then
        organizes the data in each appropriate array.
        All data is read in order so that targetarr[0] is
        the target for the data in inputarr[0].
        """
        #  Get all target labels.
        targets = re.split('\n', self.label_raw_data)
        for i in range(len(targets)):
            for k in range(10):
                if k == int(targets[i]):
                    self.targetarr[i][k] = 1
                else:
                    self.targetarr[i][k] = 0


            #if int(targets[i]) == 0:
            #    self.targetarr[i] =
            #self.targetarr[i] = int(targets[i])
        #  Get all data inputs.
        inputs = re.split('\n', self.data_raw_data)
        i = 0
        for input in inputs:
            numarr = re.split(',', input)
            x = 0
            for num in numarr:
                #  Take each number and put in numpy array.
                self.inputarr[i][x] = int(num)
                x += 1
            i += 1

    def normalize(self):
        """
        Normalize the input data from 0 to 1.
        Only use when the values considered are being weighted
        the same.
        """
        self.inputarr = self.inputarr / self.numpixels

    def get_targets(self):
        """
        :return: target array.
        """
        return self.targetarr

    def get_inputs(self):
        """
        :return: input array.
        """
        return self.inputarr

    def set_file_path(self, new_path):
        """
        :param new_path: new file path to read data from.
        """
        self.file_path = new_path
