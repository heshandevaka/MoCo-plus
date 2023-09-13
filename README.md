# MoCo-plus
Code for implementing MoCo+  (ICASSP 2024)

## Setting up

We use the [LibMTL](https://github.com/median-research-group/LibMTL.git) for running the experiments. Follow the instructions in [here](https://github.com/heshandevaka/MoCo-plus/tree/main/LibMTL) to install the LibMTL environemnt. Note that, instead of using the LibMTL folder provided in the original source repo, use the LibMTL folder provided with this repo, which includes implementation of MoCo+. You will also have to include the datasets for Office-31 and Office-home tasks before running experiments. Instructions to setting up the office tasks is provided [here](https://github.com/heshandevaka/MoCo-plus/tree/main/LibMTL/examples/office). 

## Running experiments
To run experiments with MoCo+ using Office-31, navigate to `LibMTL/examples/office` and run 
   ```shell
   ./run_mocoplus_office-31.sh
   ```
To run experiments with MoCo+ using Office-home, navigate to `LibMTL/examples/office` and run 
   ```shell
   ./run_mocoplus_office-home.sh
   ```
