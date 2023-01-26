# Requirements for the input datasets

I've based this on `loadData()` in `secondary.cpp`, which means these requirements are such that the program won't crash loading the data.

As for algorithmic constraints, that depends on the ML algorithm used, and I haven't got a lot of knowledge about that one. But I assume that all floats (see below) have to be within 0-1 (inclusive), since they appear to be representing pixel values. Well, either that or 0-255, but that would be weird because then they could be using integer values.

In general, every dataset is expected to consist of four files:
- `train_data`: The data (images) for the training set. Is a set of `N` images, of `W`x`H` pixels. Each pixel has `P` floats that make it up (e.g., `P = 1` would be greyscale, `P = 3` is RGB). Or, put differently, it is a file of `N` * `W` * `H` * `P` floats.
- `train_labels`: The golden labels for the training set. It is a set of `N` labels, each consists of `L` floats (where `L` is the size of the last layer in the network).
- `test_data`: Same as `train_data` but for the testing set.
- `test_labels`: Same as `train_labels` but for the testing set.

These datasets are looked for in the `files/` folder. However, note that there's a catch; there are three parties involved (`A`, `B` and `C`) so you will have to split your data in three to emulate that.

Concretely, the program expects the following files:
- `files/train_data_A`
- `files/train_data_B`
- `files/train_data_C`
- `files/train_labels_A`
- `files/train_labels_B`
- `files/train_labels_C`
- `files/test_data_A`
- `files/test_data_B`
- `files/test_data_C`
- `files/test_labels_A`
- `files/test_labels_B`
- `files/test_labels_C`


These parameters differ per dataset, as seen at the top of `loadData()`. I've noted down below which paramters are given for which dataset, and how to compute them from which constant so you know which constant to change to change these properties.

## MNIST
- `train_data`
  - `N` from `TRAINING_DATA_SIZE`: 8
  - `W` from `sqrt(INPUT_SIZE / 1)`: 28
  - `H` from `sqrt(INPUT_SIZE / 1)`: 28
  - `P` from `sqrt(INPUT_SIZE / 1)`: 1
  - (Note, `W`, `H` and `P` are mostly actually taken from `MNIST_LeNet.ipynb` in `scripts/`)
- `train_labels`:
  - `N` from `TRAINING_DATA_SIZE`: 8
  - `L` from `LAST_LAYER_SIZE`: 10
- Idem for `test_data` and `test_labels`, respectively, except that:
  - `N` from `TEST_DATA_SIZE`: 8

## CIFAR10
- `train_data`
  - `N` from `TRAINING_DATA_SIZE`: 8
  - `W` from `INPUT_SIZE` (value 1 in product): 33 for AlexNet, 32 for VGG16(?)
  - `H` from `INPUT_SIZE` (value 2 in product): 33 for AlexNet, 32 for VGG16(?)
  - `P` from `INPUT_SIZE` (value 3 in product): 3
- `train_labels`:
  - `N` from `TRAINING_DATA_SIZE`: 8
  - `L` from `LAST_LAYER_SIZE`: 10
- Idem for `test_data` and `test_labels`, respectively, except that:
  - `N` from `TEST_DATA_SIZE`: 8

## ImageNet
- `train_data`
  - `N` from `TRAINING_DATA_SIZE`: 8
  - `W` from `INPUT_SIZE` (value 1 in product): 56 for AlexNet, 64 for VGG16(???)
  - `H` from `INPUT_SIZE` (value 2 in product): 56 for AlexNet, 64 for VGG16(???)
  - `P` from `INPUT_SIZE` (value 3 in product): 3
- `train_labels`:
  - `N` from `TRAINING_DATA_SIZE`: 8
  - `L` from `LAST_LAYER_SIZE`: 200
- Idem for `test_data` and `test_labels`, respectively, except that:
  - `N` from `TEST_DATA_SIZE`: 8

## Tips
- Note that based on these values, the program appears to only every use the first 8 images specified in each file. I think it makes sense to increase that to more sensible, larger (and better profilable, probably) values. But note that that means you have to have at least that many images in your datasets.
