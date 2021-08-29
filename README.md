# Deep learning with PyTorch

Code for "Deep learning with PyTorch Book " from Packt.

Note: The code was built for PyTorch 0.4, so some of them may not work anymore.

<p align="center">
  <img src="images/DLwithPyTorch.jpg">
</p>

-----

## Install Pytorch

```sh
conda install pytorch==1.0.0 torchvision==0.2.1 cuda80 -c pytorch

conda install pytorch torchvision cuda80 -c soumith

# others
pip install torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html

# uninstall
pip uninstall torchvision
```

result:

```
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /home/cg/tools/anaconda3

  added / updated specs:
    - cuda80
    - pytorch
    - torchvision


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    conda-4.9.2                |   py37h06a4308_0         2.9 MB
    cuda80-1.0                 |                0           5 KB  soumith
    ninja-1.10.2               |   py37hff7bd54_0         1.4 MB
    pytorch-1.0.0              |py3.7_cuda9.0.176_cudnn7.4.1_1       498.7 MB  soumith
    torchvision-0.2.1          |             py_2          37 KB  soumith
    ------------------------------------------------------------
                                           Total:       503.0 MB

The following NEW packages will be INSTALLED:

  cuda80             soumith/linux-64::cuda80-1.0-0
  ninja              pkgs/main/linux-64::ninja-1.10.2-py37hff7bd54_0
  pytorch            soumith/linux-64::pytorch-1.0.0-py3.7_cuda9.0.176_cudnn7.4.1_1
  torchvision        soumith/noarch::torchvision-0.2.1-py_2

The following packages will be UPDATED:

  conda              conda-forge::conda-4.8.3-py37hc8dfbb8~ --> pkgs/main::conda-4.9.2-py37h06a4308_0


Proceed ([y]/n)? 


Downloading and Extracting Packages
conda-4.9.2          | 2.9 MB    | ######################################################################################################################################################################### | 100% 
torchvision-0.2.1    | 37 KB     | ######################################################################################################################################################################### | 100% 
cuda80-1.0           | 5 KB      | ######################################################################################################################################################################### | 100% 
ninja-1.10.2         | 1.4 MB    | ######################################################################################################################################################################### | 100% 
pytorch-1.0.0        | 498.7 MB  |                                                                                                                                                                           |   0% 

('Connection broken: IncompleteRead(16378 bytes read, 6 more expected)', IncompleteRead(16378 bytes read, 6 more expected))
```
