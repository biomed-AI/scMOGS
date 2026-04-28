![](Figures/Pipeline.png)

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
pip install -r requirements.txt
~~~

### Requirements
- `numpy==1.22.3`
- `pandas==1.5.3`
- `anndata==0.8.0`
- `matplotlib==3.5.1`
- `scipy==1.9.1`
- `seaborn==0.11.2`
- `scikit-learn==1.1.2`
- `torch==2.4.1`
- `torch-geometric==2.1.0.post1`
- `torchmetrics==0.9.3`
- `tqdm==4.64.0`
- `scanpy==1.9.1`
- `torch_sparse==0.6.18`
- `torch_scatter==2.1.2`
- `leidenalg==0.10.2`

## HGNN training
The train_model.py script is the core of the pipeline, training a Heterogeneous Graph Neural Network (HGNN) to integrate single-cell multi-omics data and otuput embedding for cell, gene and peak.
~~~shell
python train_model.py \
  --lr 0.0005 \
  --labsm 0.1 \
  --wd 0.1 \
  --nlayers 3 \
  --n_hid 128 \
  --nheads 8 \
  --cell_size 50 \
  --epochs_p1 100 \
  --epochs_p2 50 \
  --device 0 \
  --input_file <path_to_your_input_directory> \
  --output_file <path_to_output_directory>
~~~

