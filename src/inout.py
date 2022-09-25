import csv
import json


def read_csv(file_name):
    """
    Reads the data from the csv file and returns the tuples and attributes.
    The file should have the attributes in the first row and the tuples in the
    second row.
    :param file_name: the file name of the csv file located in the data folder
    :return: the tuples and attributes of the csv file
    """
    lines = []
    with open(f'data/{file_name}', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        attributes = next(csv_reader)  # get the attributes

        # store the lines in a list
        for line in csv_reader:
            lines.append(line)

    return lines, attributes


def read_json(file_name):
    """
    Reads the attribute types, ordinal attribute values and attributes to ignore
    from a json file with file name and returns them.
    :param file_name: the name of the json file located in the data folder
    :return: the attribute types, ordinal attribute values, attributes to ignore,
    and the decision attribute along with its value
    """
    f = open(f'data/{file_name}', 'r')  # open file
    json_dict = json.load(f)  # convert to dict

    # get the attribute types
    attribute_types = json_dict['attribute_types']
    # get ranked values per ordinal attribute
    ordinal_attribute_values = json_dict['ordinal_attribute_values']
    # get attributes to be ignored
    attributes_to_ignore = json_dict['attributes_to_ignore']
    # get the decision attribute along with value
    decision_attribute = json_dict['decision_attribute']

    return attribute_types, ordinal_attribute_values, attributes_to_ignore, decision_attribute


def read_data(json_file_name, csv_file_name):
    """
    Reads the data from the csv file and the json file and returns both its
    return values
    :param json_file_name: the name of the json file
    :param csv_file_name: the name of the csv file
    :return: all the return values of the read_csv and read_json functions
    """
    # read the attribute types and ranked values from the json file
    attribute_types, ordinal_attribute_values, attributes_to_ignore, decision_attribute = read_json(
        json_file_name)

    # read the data from the csv file
    all_tuples, attributes = read_csv(csv_file_name)

    return all_tuples, attributes, attribute_types, ordinal_attribute_values, attributes_to_ignore, decision_attribute


if __name__ == '__main__':
    attribute_types, ordinal_attribute_values, attributes_to_ignore, decision_attribute = read_json(
        'german_credit_data.json')
    print(attribute_types)
    print(ordinal_attribute_values)
    print(attributes_to_ignore)
    print(decision_attribute)

    all_tuples, attributes = read_csv('german_credit_data.csv')
    print(all_tuples)
    print(attributes)
