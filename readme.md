# DCT: A Conditional Independence Test in the Presence of Discretization
This repository contains implementation for paper : **A Conditional Independence Test in the Presence of Discretization** [\[ICLR 2025\]](https://arxiv.org/abs/2404.17644) and paper **A Sample Efficient Conditioal Independence Test in the Presence of Discretization** [\[ICML 2025\]](https://icml.cc/virtual/2025/poster/43894)

DCT and DCT-GMM are conditional independence tests specifically designed for the scenario in which only discretized versions of variables are available. 

## How to Install Required Packages 
run the code 

`conda env create -f environment.yml`

Then you will have a conda environment named 'causal'. You can further activate the environment by running

`conda activate causal`

## How to Use 

We provide two examples of running the tests in `example_cit.ipynb` and running the PC algorithm with DCT and DCT-GMM as the CI tests in `example_to_use_pc.ipynb`.

## Where is the Core Algorithm Implemented?
Core algorithm of DCT is implemented at `causal_learn.causallearn.utils.DisTestUtil.py`.

Core algorithm of DCT-GMM is implemented at `causal_learn.causallearn.utils.DisTestGMMUtil.py`.
