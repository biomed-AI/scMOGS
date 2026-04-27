![](Figures/Pipeline1.png)

# scMOGS: a cross-modality attention heterogeneous graph neural network integrating single-cell multi-omics data for elucidating cell-disease association 
scMOGS is a model that integrates single-cell multi-omics data with GWAS summary statistics for evaluating variant impact on cells and disease-cell associations.

## Installation

To reproduce **scMOGS**, we suggest first creating a conda environment by:

~~~shell
conda create -n scMOGS python=3.9
conda activate scMOGS
~~~

and then run the following code to install the required package:

~~~shell
cd scMOGS-master
pip install --editable ./
~~~

### Requirements
- `PyTorch version >= 1.5.0`
- `Python version >= 3.6`
