import json

import pandas as pd

from process import make_numeric


class Read:
    def __init__(self, f_name, f_type):
        self.f_name = f_name
        self.f_type = f_type

    def read(self):
        if self.f_type == 'csv':
            return self._read_csv()
        elif self.f_type == 'json':
            return self._read_json()
        else:
            raise Exception('File type not supported')

    def _read_csv(self):
        """
        Reads the data from the csv file and determines the tuples and
        attributes. The file should have the attributes in the first row and the
        tuples in the second row.
        :return: nothing
        """
        self.df = pd.read_csv(self.f_name)

    def _read_json(self):
        """
        Reads the attribute types, ordinal attribute values and attributes to ignore
        from a json file with file name and returns them.
        :return: nothing
        """
        f = open(self.f_name, 'r')  # open file
        json_dict = json.load(f)  # convert to dict

        # get the attribute types
        self.attribute_types = json_dict['attribute_types']

        # get ranked values per ordinal attribute
        self.ordinal_attribute_values = json_dict['ordinal_attribute_values']

        # get attributes to be ignored
        self.attributes_to_ignore = json_dict['attributes_to_ignore']

        # get the decision attribute along with value
        self.decision_attribute = json_dict['decision_attribute']

        # get the unknown list
        self.unknowns_list = []
        if 'unknowns' in json_dict:
            self.unknowns_list = json_dict['unknowns']


def read_data(json_file_name, csv_file_name):
    """
    Reads the data from the csv file and the json file and returns both its
    return values
    :param json_file_name: the name of the json file
    :param csv_file_name: the name of the csv file
    :return: all the return values of the read_csv and read_json functions
    """
    # read the data from the csv file
    r = Read(csv_file_name, 'csv')
    r.read()
    all_tuples = r.df

    # read the attribute types and ranked values from the json file
    r = Read(json_file_name, 'json')
    r.read()
    r.df = all_tuples

    return r


if __name__ == '__main__':
    r = Read('data/german_credit_data.json', 'json')
    r.read()
    print(r.attribute_types)
    print(r.ordinal_attribute_values)
    print(r.attributes_to_ignore)
    print(r.decision_attribute)

    r = Read('data/german_credit_data.csv', 'csv')
    r.read()
    all_tuples = r.df

    df = make_numeric(all_tuples)
    print(all_tuples.Age.dtype)
    print(all_tuples)
