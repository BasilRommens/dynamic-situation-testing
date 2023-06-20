# Dynamic situation testing

A web tool based on the paper
by [Luong et al.](https://dl.acm.org/doi/pdf/10.1145/2020408.2020488)
to create a proper understanding of discrimination using situation testing using
the Data Context Map as a visualization method
from [Cheng and Mueller](https://doi.org/10.1109/TVCG.2015.2467552).
This is the 2nd research project at the University of Antwerp in the master of
computer science.

## Setup

The base version of python required is 3.10.8 and currently only works on Linux.
To set up the project for the first time run the following command in terminal
from the root of the project.

```shell
./setup.sh
```

## Run

To run the project run the following command in terminal from the root of the
project. It will also open a browser tab with the correct url. The german credit
data set with random class labels is used as default.

```shell
./run.sh
```

## Data

The data is stored in the `data` folder, two files are expected per data set.
One is the csv file containing the tuples and the second one more information
about the data types.

### CSV file

The tuples should be stored in a csv format, no special format required. The
only requirement for this data is that all the unknown values should be marked
with `NA`, so that they can be properly converted.

#### CSV example

```csv
age,gender,work,class
12,m,1,0
30,f,1,1
20,m,NA,0
```

### JSON file

In the json file there are more directives as how to interpret the data and how
to process it. There are 3 different
sections parts in this file:

1. `attribute_types`: For each attribute the type is specified. The types
   are `interval`, `nominal` and `ordinal`.
2. `ordinal_attribute_values`: For each ordinal attribute the values are
   specified. As key, we have all the possible values of the ordinal attribute
   and as value the order their order. `0` is the lowest in the ordering and the
   increase is by 1.
3. `attributes_to_ignore`: These are the attributes that you don't take into
   account when processing the distances between the tuples. They can be chosen
   by specifying the name of the attribute.
4. `decision_attribute`: A combination of the decision attribute and the
   decision value. This is for basing the discrimination score on.
5. `unknowns`: This is a list of values for attributes that are unknown.

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
  ],
  "decision_attribute": {
    "Class": 0
  },
  "unknowns": [
    "NA"
  ]
}
```

### Included data sets
A random labeled german credit dataset is present in this repository. Together
with a corresponding json file. The json files of the COMPAS and Adult data sets
are also included. An example CSV and JSON file are present.
