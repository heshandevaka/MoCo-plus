## Office-31 and Office-Home

The Office-31 dataset [[1]](#1) consists of three classification tasks on three domains: Amazon, DSLR, and Webcam, where each task has 31 object categories. It can be download [here](https://www.cc.gatech.edu/~judy/domainadapt/#datasets_code). This dataset contains 4,110 labeled images and we randomly split these samples, with 60% for training, 20% for validation, and the rest 20% for testing. 

The Office-Home dataset [[2]](#2) has four classification tasks on four domains: Artistic images (abbreviated as Art), Clip art, Product images, and Real-world images. It can be download [here](https://www.hemanthdv.org/officeHomeDataset.html). This dataset has 15,500 labeled images in total and each domain contains 65 classes. We divide the entire data into the same proportion as the Office-31 dataset. 

Both datasets belong to the multi-input setting in MTL. Thus, the ``multi_input`` must be ``True`` for both of the two office datasets.

We use the ResNet-18 network pretrained on the ImageNet dataset followed by a fully connected layer as a shared encoder among tasks and a fully connected layer is applied as a task-specific output layer for each task. All the input images are resized to <img src="https://render.githubusercontent.com/render/math?math=3\times224\times224">.

### Run a Model

The script ``train_office.py`` is the main file for training and evaluating a MTL model on the Office-31 or Office-Home dataset. A set of command-line arguments is provided to allow users to adjust the training parameter configuration. 

Some important  arguments are described as follows.

- ``weighting``: The weighting strategy. Refer to [here](../../LibMTL#supported-algorithms).
- ``arch``: The MTL architecture. Refer to [here](../../LibMTL#supported-algorithms).
- ``gpu_id``: The id of gpu. The default value is '0'.
- ``seed``: The random seed for reproducibility. The default value is 0.
- ``optim``: The type of the optimizer. We recommend to use 'adam' here.
- ``dataset``: Training on Office-31 or Office-Home. Options: 'office-31', 'office-home'.
- ``dataset_path``: The path of the Office-31 or Office-Home dataset.
- ``bs``: The batch size of training, validation, and test data. The default value is 64.

The complete command-line arguments and their descriptions can be found by running the following command.

```shell
python train_office.py -h
```

If you understand those command-line arguments, you can train a MTL model by running a command like this. 

```shell
python train_office.py --weighting WEIGHTING --arch ARCH --dataset_path PATH --gpu_id GPU_ID --multi_input
```

### References

<a id="1">[1]</a> Kate Saenko, Brian Kulis, Mario Fritz, and Trevor Darrell. Adapting visual category models to new domains. In *Proceedings of the 6th European Conference on Computer Vision*, 213–226. 2010.

<a id="2">[2]</a> Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing network for unsupervised domain adaptation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 5018–5027. 2017.