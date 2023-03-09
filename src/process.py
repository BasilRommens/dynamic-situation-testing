import numpy as np
from sklearn import preprocessing


def convert_value(value, attribute, attribute_type, ordinal_attribute_values):
    """
    convert the value to the correct type
    :param value: the value to convert
    :param attribute: the attribute of the value
    :param attribute_type: the type of the value
    :param ordinal_attribute_values: the ordinal attribute values for ordinal
    conversion
    :return: the converted value
    """
    # if no value is given return None
    if type(value) == np.cfloat and np.isnan(value):
        return np.nan
    if type(value) == float and np.isnan(value):
        return np.nan

    # determine the value to return based on the attribute type
    if attribute_type == 'interval':
        return int(value)
    elif attribute_type == 'nominal':
        return value
    elif attribute_type == 'ordinal':
        if type(value) != str:
            value = str(value)
        # get the ranked values for the ordinal attribute
        return int(ordinal_attribute_values[attribute][value])


def remove_redundant_attributes(all_tuples, attributes_to_ignore):
    """
    removes the attributes that are in the attributes_to_ignore list, this also
    happens in the all tuples.
    :param all_tuples: all the tuples to remove the attributes from
    :param attributes: the attribute list to minimize
    :param attributes_to_ignore: the attributes to ignore
    :return: the tuples and attributes without the redundant attributes
    """
    all_tuples = all_tuples.drop(columns=attributes_to_ignore)

    return all_tuples


def process_tuples(all_tuples, attribute_types,
                   ordinal_attribute_values, attributes_to_ignore):
    """
    process the tuples to the correct types and remove their attributes to
    ignore and return the processed attribute list
    :param all_tuples: all the tuples to process
    :param attributes: the attributes of the tuples
    :param attribute_types: the types of all the attributes
    :param ordinal_attribute_values: the ordinal attribute values
    :param attributes_to_ignore: attributes to ignore
    :return: the processed tuples and attributes
    """
    # remove the redundant attributes
    all_tuples = remove_redundant_attributes(all_tuples, attributes_to_ignore)
    attributes = list(all_tuples.columns)

    # convert the values to the correct type
    for tuple_idx in all_tuples.index:
        new_tuple = list()
        for idx, attribute in enumerate(attributes):
            tuple_value = all_tuples[attribute][tuple_idx]
            # get type per attribute
            attribute_type = attribute_types[attribute]

            # convert the value to the correct type
            new_value = convert_value(tuple_value, attribute, attribute_type,
                                      ordinal_attribute_values)
            new_tuple.append(new_value)

        all_tuples.iloc[tuple_idx] = new_tuple

    return all_tuples, attributes


def process_ranked_values(ordinal_attribute_values):
    """
    process the ordinal attribute values to ranked values
    :param ordinal_attribute_values: the ordinal attribute values
    :return: the ranked values per ordinal attribute
    """
    ranked_values = dict()
    for attribute, attribute_values in ordinal_attribute_values.items():
        ranked_values[attribute] = list(attribute_values.values())

    return ranked_values


def process_decision_attribute(attribute_types, decision_attribute,
                               ordinal_attribute_values):
    """
    process the decision attribute to the correct type
    :param attribute_types: the types of the attributes
    :param decision_attribute: the decision attribute
    :param ordinal_attribute_values: the ordinal attribute values
    :return: the processed decision attribute
    """
    decision_attribute_name = list(decision_attribute.keys())[0]
    decision_attribute_value = decision_attribute[decision_attribute_name]
    decision_attribute_type = attribute_types[decision_attribute_name]

    # convert the value to the correct type
    decision_attribute_value = convert_value(decision_attribute_value,
                                             decision_attribute_name,
                                             decision_attribute_type,
                                             ordinal_attribute_values)

    return {decision_attribute_name: decision_attribute_value}


def process_all(r):
    """
    process all the tuples, create the ranked values and return attributes that
    aren't ignored
    :param r: read object with all the data to be processed
    :return: the processed tuples, ranked values, attributes and decision
    attribute
    """
    all_tuples = r.df
    attribute_types = r.attribute_types
    ordinal_attribute_values = r.ordinal_attribute_values
    attributes_to_ignore = r.attributes_to_ignore
    decision_attribute = r.decision_attribute

    # process the tuples
    tuples, attributes = process_tuples(all_tuples, attribute_types,
                                        ordinal_attribute_values,
                                        attributes_to_ignore)

    # process the ordinal attribute values to ranked values
    ranked_values = process_ranked_values(ordinal_attribute_values)
    # process decision attribute
    decision_attribute = process_decision_attribute(attribute_types,
                                                    decision_attribute,
                                                    ordinal_attribute_values)

    return tuples, ranked_values, decision_attribute


def make_numeric(df, string_to_value_dict=dict()):
    """
    Converts all the values in numeric values instead of strings
    :param df: the dataframe containing
    :param string_to_value_dict: a dictionary containing the attributes that
    have a numerical value
    :return: the dataframe with solely numeric values
    """
    for col in df.columns:
        if df[col].dtype != 'object':
            continue

        # use the dict if the column is an attribute has a numerical value
        if col in string_to_value_dict.keys():
            # convert the string attributes to numeric values
            string_to_value = string_to_value_dict[col]
            df[col] = df[col].map(string_to_value)
        else:  # otherwise we just use a label encoder
            le = preprocessing.LabelEncoder()
            df[col] = le.fit_transform(df[col])
    df.fillna(-1, inplace=True)

    # convert all the types of the columns to floats
    for col in df.columns:
        df[col] = df[col].astype(np.float64)

    return df


if __name__ == "__main__":
    import inout

    json_file_name = 'german_credit_data.json'
    csv_file_name = 'german_credit_data_class.csv'
    # open the files
    r = inout.read_data(json_file_name, csv_file_name)

    # process the data
    tuples, ranked_values, decision_attribute = process_all(r)
    print(tuples)
    print(ranked_values)
    print(decision_attribute)
