import numpy as np


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
    if value == 'NA':
        return None

    # determine the value to return based on the attribute type
    if attribute_type == 'interval':
        return int(value)
    elif attribute_type == 'nominal':
        return value
    elif attribute_type == 'ordinal':
        # get the ranked values for the ordinal attribute
        return int(ordinal_attribute_values[attribute][value])


def remove_redundant_attributes(all_tuples, attributes, attributes_to_ignore):
    """
    removes the attributes that are in the attributes_to_ignore list, this also
    happens in the all tuples.
    :param all_tuples: all the tuples to remove the attributes from
    :param attributes: the attribute list to minimize
    :param attributes_to_ignore: the attributes to ignore
    :return: the tuples and attributes without the redundant attributes
    """
    # collect the idces of all the attributes not to be ignored
    attribute_idces = list()
    for attribute in attributes:
        if attribute in attributes_to_ignore:
            continue
        attribute_idx = attributes.index(attribute)
        attribute_idces.append(attribute_idx)

    # keep only the attributes not to be ignored
    attributes = list(np.array(attributes)[attribute_idces])
    # transform the tuples to only contain the attributes not to be ignored
    all_tuples = list(
        map(lambda x: list(np.array(x)[attribute_idces]), all_tuples))

    return all_tuples, attributes


def process_tuples(all_tuples, attributes, attribute_types,
                   ordinal_attribute_values, attributes_to_ignore):
    """
    process the tuples to the correct types and remove their attributes to
    ignore and return the processed attribute list too
    :param all_tuples: all the tuples to process
    :param attributes: the attributes of the tuples
    :param attribute_types: the types of all the attributes
    :param ordinal_attribute_values: the ordinal attribute values
    :param attributes_to_ignore: attributes to ignore
    :return: the processed tuples and attributes
    """
    # remove the redundant attributes
    all_tuples, attributes = remove_redundant_attributes(all_tuples, attributes,
                                                         attributes_to_ignore)

    # convert the values to the correct type
    tuples = list()
    for tuple in all_tuples:
        new_tuple = []
        for idx, value in enumerate(tuple):
            # get type per attribute
            attribute = attributes[idx]
            attribute_type = attribute_types[attribute]

            # convert the value to the correct type
            new_value = convert_value(value, attribute, attribute_type,
                                      ordinal_attribute_values)

            new_tuple.append(new_value)

        tuples.append(new_tuple)

    return tuples, attributes


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


def process_all(all_tuples, attributes, attribute_types,
                ordinal_attribute_values, attributes_to_ignore,
                decision_attribute):
    """
    process all the tuples, create the ranked values and return attributes that
    aren't ignored
    :param all_tuples: all the tuples to process
    :param attributes: their corresponding attributes
    :param attribute_types: the types of the attributes
    :param ordinal_attribute_values: the ordinal attribute values
    :param attributes_to_ignore: the attributes to ignore
    :param decision_attribute: the decision attribute
    :return: the processed tuples, ranked values, attributes and decision
    attribute
    """
    # process the tuples
    tuples, attributes = process_tuples(all_tuples, attributes, attribute_types,
                                        ordinal_attribute_values,
                                        attributes_to_ignore)

    # process the ordinal attribute values to ranked values
    ranked_values = process_ranked_values(ordinal_attribute_values)
    # process decision attribute
    decision_attribute = process_decision_attribute(attribute_types,
                                                    decision_attribute,
                                                    ordinal_attribute_values)

    return tuples, ranked_values, attributes, decision_attribute


if __name__ == "__main__":
    import inout

    json_file_name = 'german_credit_data.json'
    csv_file_name = 'german_credit_data_class.csv'
    # open the files
    all_tuples, attributes, attribute_types, ordinal_attribute_values, attributes_to_ignore, decision_attribute = inout.read_data(
        json_file_name, csv_file_name)

    # process the data
    tuples, ranked_values, attributes, decision_attribute = process_all(
        all_tuples, attributes,
        attribute_types,
        ordinal_attribute_values,
        attributes_to_ignore,
        decision_attribute)
    print(tuples)
    print(ranked_values)
    print(attributes)
    print(decision_attribute)
