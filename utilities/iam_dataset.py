import os
import urllib
import sys
import time
import glob
import pickle
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from mxnet.gluon.data import dataset

class IAMDataset(dataset.ArrayDataset):
    
    """
    Parameters
    ----------
    parse_method: str, Required
        To select the method of parsing the images of the passage
        Available options: [form, form_bb, line, word]
    
    root: str, default: dataset/iamdataset
        Location to save the database

    train: bool, default True
        Whether to load the training or testing set of writers.

    output_data_type: str, default text
        What type of data you want as an output: Text or bounding box.
        Available options are: [text, bb]
     
    output_parse_method: str, default None
        If the bounding box (bb) was selected as an output_data_type, 
        this parameter can select which bb you want to obtain.
        Available options: [form, line, word]
        
    output_form_text_as_array: bool, default False
        When output_data is set to text and the parse method is set to form or form_original,
        if output_form_text_as_array is true, the output text will be a list of lines string
    """
    
    MAX_IMAGE_SIZE_FORM = (1120, 800)
    MAX_IMAGE_SIZE_LINE = (60, 800)
    MAX_IMAGE_SIZE_WORD = (30, 140)
    def __init__(self, parse_method,
                 root=os.path.join(os.path.dirname(__file__), '..', 'dataset', 'iamdataset'), 
                 train=True, output_data="text",
                 output_parse_method=None,
                 output_form_text_as_array=False):

        _parse_methods = ["form", "form_original", "form_bb", "line", "word"]
        error_message = "{} is not a possible parsing method: {}".format(
            parse_method, _parse_methods)
        assert parse_method in _parse_methods, error_message
        self._parse_method = parse_method
        
        
        self._train = train

        _output_data_types = ["text", "bb"]
        error_message = "{} is not a possible output data: {}".format(
            output_data, _output_data_types)
        assert output_data in _output_data_types, error_message
        self._output_data = output_data

        if self._output_data == "bb":
            assert self._parse_method in ["form", "form_bb"], "Bounding box only works with form."
            _parse_methods = ["form", "line", "word"]
            error_message = "{} is not a possible output parsing method: {}".format(
                output_parse_method, _parse_methods)
            assert output_parse_method in _parse_methods, error_message
            self._output_parse_method = output_parse_method
            
            #self.image_data_file_name = os.path.join(root, "image_data-{}-{}-{}*.h5".format(
            #    self._parse_method, self._output_data, self._output_parse_method))
            #self.image_data_file_name = os.path.join(root, "image_data-{}-{}-{}*.plk".format(
            #    self._parse_method, self._output_data, self._output_parse_method))
        #else:
            #self.image_data_file_name = os.path.join(root, "image_data-{}-{}*.h5".format(self._parse_method, self._output_data))
        #    self.image_data_file_name = os.path.join(root, "image_data-{}-{}*.plk".format(self._parse_method, self._output_data))

        self._root = root
        #if not os.path.isdir(root):
        #    os.makedirs(root)
        self._output_form_text_as_array = output_form_text_as_array
        
        data = self._get_data()
        super(IAMDataset, self).__init__(data)
        
    def _get_data(self):      
        
        '''
        Function to get the data and to extract the data for training or testing
        
        Returns
        -------
        pd.DataFrame
            A dataframe (subject, image, and output) that contains only the training/testing data

        '''
        
        print("Get data")

        #if len(glob.glob(self.image_data_file_name)) > 0:
        #    logging.info("Loading data from pickle")
        #    images_data = self._load_dataframe_chunks(self.image_data_file_name)
        #else:
        images_data = self._process_data()

        # Extract train or test data out
        train_subjects, test_subjects = self._process_subjects()
        if self._train:
            data = images_data[np.in1d(self._convert_subject_list(images_data["subject"]),
                                       train_subjects)]
        else:
            data = images_data[np.in1d(self._convert_subject_list(images_data["subject"]),
                                       test_subjects)]
        return data
    
    def _process_data(self):
        ''' 
        Function that iterates through the downloaded xml file to gather the input images and the
        corresponding output.
        
        Returns
        -------
        pd.DataFrame
            A pandas dataframe that contains the subject, image and output requested.
        '''
        image_data = []
        xml_files = glob.glob(self._root + "/xml/*.xml")
        print("Processing data:")
        logging.info("Processing data")
        for i, xml_file in enumerate(xml_files):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            height, width = int(root.attrib["height"]), int(root.attrib["width"])
            for item in root.iter(self._parse_method.split("_")[0]):
                # Split _ to account for only taking the base "form", "line", "word" that is available in the IAM dataset
                if self._parse_method in ["form", "form_bb", "form_original"]:
                    image_id = item.attrib["id"]
                else:
                    tmp_id = item.attrib["id"]
                    tmp_id_split = tmp_id.split("-")
                    image_id = os.path.join(tmp_id_split[0], tmp_id_split[0] + "-" + tmp_id_split[1], tmp_id)
                image_filename = os.path.join(self._root, self._parse_method.split("_")[0], image_id + ".png")
                image_arr = self._pre_process_image(image_filename)
                if image_arr is None:
                    continue
                output_data = self._get_output_data(item, height, width)
                #if self._parse_method == "form_bb":
                #    image_arr, output_data = self._crop_and_resize_form_bb(item, image_arr, output_data, height, width)
                image_data.append([item.attrib["id"], image_arr, output_data])
        image_data = pd.DataFrame(image_data, columns=["subject", "image", "output"])
        #self._save_dataframe_chunks(image_data, self.image_data_file_name)
        return image_data    

    def _pre_process_image(self, img_in):
        '''
        Function to read the image and convert it to array data
        It also resizes the image to standard size if it is of type ["form","form_bb","line","word"]
        
        Parameters
        ----------
        img_in: str, Required
        Contains the filename of the image
        
        Returns
        ----------
        img_arr
        An image converted to an array
        '''
        #print("Pre Processing Image")
        im = cv2.imread(img_in, cv2.IMREAD_GRAYSCALE)
        if np.size(im) == 1: # skip if the image data is corrupt.
            return None
        # reduce the size of form images so that it can fit in memory.
        if self._parse_method in ["form", "form_bb"]:
            im, _ = None #resize_image(im, self.MAX_IMAGE_SIZE_FORM) #resize function to be done soon
        if self._parse_method == "line":
            im, _ = None #resize_image(im, self.MAX_IMAGE_SIZE_LINE)
        if self._parse_method == "word":
            im, _ = None #resize_image(im, self.MAX_IMAGE_SIZE_WORD)
        img_arr = np.asarray(im)
        return img_arr 

    def _get_output_data(self, item, height, width):
        
        ''' 
        Function to obtain the output data (both text and bounding boxes).
        Note that the bounding boxes are rescaled based on the rescale_ratio parameter.

        Parameter
        ---------
        item: xml.etree 
            XML object for a word/line/form.

        height: int
            Height of the form to calculate percentages of bounding boxes

        width: int
            Width of the form to calculate percentages of bounding boxes

        Returns
        -------

        np.array
            A numpy array of the output requested (text or the bounding box)
        '''
        
        #print("Get output data")
        output_data = []
        if self._output_data == "text":
            if self._parse_method in ["form", "form_bb", "form_original"]:
                text = ""
                for line in item.iter('line'):
                    text += line.attrib["text"] + "\n"
                output_data.append(text)
            else:
                output_data.append(item.attrib['text'])
        else:
            for item_output in item.iter(self._output_parse_method):
                bb = self._get_bb_of_item(item_output, height, width)
                if bb == None: # Account for words with no letters
                    continue
                output_data.append(bb)
        output_data = np.array(output_data)
        return output_data    

    def _process_subjects(self, train_subject_lists = ["trainset", "validationset1", "validationset2"],
                          test_subject_lists = ["testset"]):
        
        ''' 
        Function to organise the list of subjects to training and testing.
        The IAM dataset provides 4 files: trainset, validationset1, validationset2, and testset each
        with a list of subjects.
        
        Parameters
        ----------
        
        train_subject_lists: [str], default ["trainset", "validationset1", "validationset2"]
            The filenames of the list of subjects to be used for training the model

        test_subject_lists: [str], default ["testset"]
            The filenames of the list of subjects to be used for testing the model

        Returns
        -------

        train_subjects: [str]
            A list of subjects used for training

        test_subjects: [str]
            A list of subjects used for testing
        '''
      
        print("Processing subjects")
        train_subjects = []
        test_subjects = []
        for train_list in train_subject_lists:
            subject_list = pd.read_csv(os.path.join(self._root, "subject", train_list+".txt"))
            train_subjects.append(subject_list.values)
        for test_list in test_subject_lists:
            subject_list = pd.read_csv(os.path.join(self._root, "subject", test_list+".txt"))
            test_subjects.append(subject_list.values)

        train_subjects = np.concatenate(train_subjects)
        test_subjects = np.concatenate(test_subjects)
        if self._parse_method in ["form", "form_bb", "form_original"]:
            new_train_subjects = []
            for i in train_subjects:
                form_subject_number = i[0].split("-")[0] + "-" + i[0].split("-")[1]
                new_train_subjects.append(form_subject_number)
            new_test_subjects = []
            for i in test_subjects:
                form_subject_number = i[0].split("-")[0] + "-" + i[0].split("-")[1]
                new_test_subjects.append(form_subject_number)
            train_subjects, test_subjects = new_train_subjects, new_test_subjects
        return train_subjects, test_subjects

    def _convert_subject_list(self, subject_list):
        
        ''' 
        Function to convert the list of subjects for the "word" parse method
        
        Parameters
        ----------
        
        subject_lists: [str]
            A list of subjects

        Returns
        -------

        subject_lists: [str]
            A list of subjects that is compatible with the "word" parse method

        '''

        print("Convert subject list")
        if self._parse_method == "word":
            new_subject_list = []
            for sub in subject_list:
                new_subject_number = "-".join(sub.split("-")[:3])
                new_subject_list.append(new_subject_number)
            return new_subject_list
        else:
            return subject_list
                
    def __getitem__(self, idx):
        return (self._data[0].iloc[idx].image, self._data[0].iloc[idx].output)
