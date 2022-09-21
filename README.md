# Dynamic situation testing

A web tool based on the paper by [Luong et al.](https://dl.acm.org/doi/pdf/10.1145/2020408.2020488) to create a proper
understanding of discrimination using situation testing. This is the 2nd research project at the university of antwerp
in the master of computer science.

## Data

The data is stored in the `data` folder, two files are expected per data set. One is the csv file containing the tuples
and the second one more information about the data types.

### CSV file

The tuples should be stored in a csv format, no special format required. The only requirement for this data is that all
the unknown values should be marked with `NA`, so that they can be converted properly.

#### CSV example

```csv
age,gender,work
12,m,1
30,f,1
20,m,1
```

### JSON file

In the json file there are more directives as how to interpret the data and how to process it. There are 3 different
sections parts in this file:

1. `attribute_types`: For each attribute the type is specified. The types are `numeric`, `categorical` and `ordinal`.
2. `ordinal_attribute_values`: For each ordinal attribute the values are specified. As key we have all the possible
   values of the ordinal attribute and as value the order their order. `0` is the lowest in the ordering and the
   increase is by 1.
3. `attributes_to_ignore`: These are the attributes that you don't take into account when processing the tuples. They
   can be chosen by specifying the name of the attribute.

#### JSON example

```json
{
  "attribute_types": {
    "Age": "interval",
    "Sex": "nominal",
    "Saving accounts": "ordinal"
  },
  "ordinal_attribute_values": {
    "Saving accounts": {
      "little": 0,
      "moderate": 1,
      "rich": 2
    }
  },
  "attributes_to_ignore": [
    "Age"
  ]
}
```