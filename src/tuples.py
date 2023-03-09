from helper import is_in_valid_tuples


def get_tuple_discrimination_type(valid_tuples, tuples):
    tuple_markers = list()
    for idx, tuple in enumerate(tuples.values):
        valid_tuple_idx = is_in_valid_tuples(valid_tuples, idx)
        if valid_tuple_idx < 0:
            tuple_markers.append('neutral')
        elif valid_tuples[valid_tuple_idx][1] == 0:
            tuple_markers.append('sensitive')
        elif valid_tuples[valid_tuple_idx][1] > 0:
            tuple_markers.append('negative discrimination')
        elif valid_tuples[valid_tuple_idx][1] < 0:
            tuple_markers.append('positive discrimination')
    return tuple_markers


def get_tuples_with_attr(df, protected_attributes):
    for attr, values in protected_attributes.items():
        for value in values:
            df = df.loc[df[attr] == value]
    return df.index.values
