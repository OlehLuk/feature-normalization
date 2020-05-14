# feature_normalization
This is a straightforward implementation for features normalization.

Currently Z-normalization is provided (`ZNormalizer`), 
other options can be implemented by implementing Normalizer interface.

`ZNormalizer` uses train file to learn sample mean and std for each feature. 
Afterwards, these values are used for normalization: (x-mean)/std

To run an example:
1. Clone the repository
2. Put your train and test files in the data folder
3. Run `test.py`. For example: 
`python3 test.py --train data/train.tsv --input data/test.tsv --output data/test_proc.tsv` or 
`python3 test.py - data/train.tsv -i data/test.tsv -o data/test_proc.tsv`

Package can also be installed with:
`pip install git+https://github.com/OlehLuk/feature-normalization.git` (or 
`pip3 install git+https://github.com/OlehLuk/feature-normalization.git` if you have both python 2 and python 3 installed)