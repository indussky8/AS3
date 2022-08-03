# The More, The Better? Active Silencing of Non-Positive Transfer for Efficient Multi-Domain Few-Shot Classification

The official implementation of our AS3 for efficient multi-domain few-shot classification.


## Dependencies
This code requires the following
* Python 3.6
* Pytorch 1.8.1
* Tensorflow 2.4.1


## Configure Meta-Dataset
* Follow the the "User instructions" in the Meta-Dataset repository (https://github.com/google-research/meta-dataset) for "Installation" and "Downloading and converting datasets". Brace yourself, the full process would take around a day.
* If you want to test out-of-domain behavior on additional datasets, namely, MNIST, CIFAR10, CIFAR100, follow the installation instructions in the CNAPs repository to get these datasets. This step is takes little time and we recommended to do it.

## Configure Bert 
Plz download the pre-trained bert model (e.g., bert-base-uncased) from (https://huggingface.co/transformers/pretrained_models.html) and put it in create_bert_embedding dir
## Usage
1. First, Plz modify META_DATASET_ROOT and META_RECORDS_ROOT in paths.py according to your Meta-Dataset dir.

2. Get the novel class of each few-shot task for each test dataset. Plz run ./data/main_novel_class.py

3. Get the bert embedding of each class, Plz run ./create_bert_embedding/create_bert_embedding.py

4. Perform the base class selection. Plz run ./run/base_class_selection/base_class_selection.py

5. To get the feature extractors, Plz run main.py and modify the corresponding parameter, including trgset, selected_dataspec_root_dir, model_save_dir and so on

## Offline Testing (optional)
To speed up the testing procedure, one could first dump the features on the hard drive, and then use them for selection directly, without needing to run a CNN. To do so, follow the steps:

1. Dump test features extracted from the test episodes on your hard drive by running ./data/main.py
Plz modify the corresponding parameters, including trgset, selected_dataspec_root_dir, model_save_dir and so on

2. Test AS3 offline. Depending on your desired feature extractor, Plz run main_test_ds3.py

## Acknowledgements
This codebase is heavily borrowed from [SUR](https://github.com/dvornikita/SUR) and [Meta-Dataset](https://github.com/google-research/meta-dataset)


