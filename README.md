Human-in-the-Loop Assisted De Novo Molecular Design
=================================================================================================================

This repository contains source code of the methods presented in paper:

I. Sundin, A. Voronov, H. Xiao, K. Papadopoulos, E. J. Bjerrum, M. Heinonen, A. Patronov, S. Kaski, O. Engkvist. (2022). Human-in-the-Loop Assisted de Novo Molecular Design. J Cheminform 14, 86 (2022). https://doi.org/10.1186/s13321-022-00667-8

After the paper was published, this repository is in read-only Archive mode. Further development on this topic will get a separate repository.

Installation
-------------

1. Install [Conda](https://conda.io/projects/conda/en/latest/index.html)
2. Clone this Git repository
3. In a shell, go to the repository and create the Conda environment, e.g.
   
        $ conda env create -f adapt-mpo.yml

4. Activate the environment:
   
        $ conda activate adapt_mpo

5. Run the method (see 'Usage')

     
System Requirements
-------------------

Adaptive MPO:
* Python 3.7
* Reinvent 3.2 (https://github.com/MolecularAI/Reinvent)
* reinvent-models 0.0.15rc1 (https://github.com/MolecularAI/reinvent-models)
* reinvent-scoring 0.0.73
* reinvent-chemistry 0.0.51
* Pystan 2.19.11
* This code has been tested on Linux

Chemist's Component: 
* Python 3.7
* Reinvent 3.2 (https://github.com/MolecularAI/Reinvent)
* reinvent-models 0.0.15rc1 (https://github.com/MolecularAI/reinvent-models)
* reinvent-scoring with gpflow [fork](https://github.com/MolecularAI/reinvent-scoring-gpflow)
* reinvent-chemistry 0.0.51
* GPflow 2.3.1
* ["Wall of molecules" (MolWall) GUI](https://github.com/MolecularAI/molwall)
* This code has been tested on Linux


Usage - Adaptive MPO
--------------------------------------------

In Task1_Adaptive_MPO_HITL.py, modify paths

```
reinvent_dir = "/path/to/Reinvent"
reinvent_env = "/path/to/conda_environment"
output_dir = "/path/to/result/directory/{}_seed{}".format(jobid,seed))
```

to match those in your system.

Add a prior agent to Reinvent:
* create a directory "data" under /path/to/Reinvent
* copy random.prior.new from [ReinventCommunity](https://github.com/MolecularAI/ReinventCommunity/tree/master/notebooks/models) to /path/to/Reinvent/data/

To run, execute: 
		
		$ python Task1_Adaptive_MPO_HITL.py acquisition seed

where acquisition is one of the following query selection strategies: random, uncertainty, greedy or thompson (see explanation below), and seed is a random seed for reproducibility and running multiple replicas with different initializations.

Supported query selection strategies:
* Random sampling: 'random'
* Uncertainty sampling: 'uncertainty'
* Pure exploitation: 'greedy'
* Thompson sampling: 'thompson'


Usage - Chemist's Component
--------------------------------------------

Chemist's component experiments consists of two phases:
1. Human-in-the-loop interaction: can be run either with a simulated human or using GUI for user interaction. 
- Files:
    * Conda environment cc_env_hitl.yml
    * Jupyter Notebook Task2_Chemists_Component.ipynb
2. Evaluating the performance of the resulting chemist's component as Reinvent scoring function. 
- Files:
    * Conda environment cc_env_reinvent.yml
    * Scripts created by Task2_Chemists_Component.ipynb in directory 'output_dir/loop0/'
    * Jupyter Notebook evaluate_results_Task2.ipynb to analyze and plot the results

Preparing the setup
- Create conda virtual environments cc_env_hitl and cc_env_reinvent from cc_env_hitl.yml and cc_env_reinvent.yml respectively
- Manual modifications to cc_env_reinvent:
	* Build and install reinvent-scoring locally from [fork](https://github.com/Augmented-Drug-Design-Human-in-the-Loop/reinvent-scoring-gpflow) to support GPflow models.
	* Copy test_config.json to the environment if needed: In reinvent-scoring-gpflow run

    		$ scp ./reinvent_scoring/configs/test_config.json /path/to/env/lib/python3.7/site-packages/reinvent_scoring/configs/test_config.json


To run:

Activate cc_env_hitl

1. Open 'Task2_Chemists_Component.ipynb' notebook. 
	- In the Configuration-cell, set id number of the experiment
	- Set paths to Reinvent (```reinvent_dir```) and the conda environment created from cc_env_reinvent.yml (```reinvent_env```) and to a directory for saving the results (```output_dir```) and create it if needed.
	- Set acquisition method: Thompson sampling: 'thompson'; random sampling: 'random'; uncertainty sampling: 'uncertainty'; or pure exploitation: 'greedy'.
	- Select between simulated chemist (```simulated_human=True```) and user-interaction via GUI (```simulated_human=False```)
2. Run the notebook; if ```simulated_human=False``` it will wait for input after writing the first query (query_it1.csv). If ```simulated_human=True```, the notebook will continue a whole run of simulated HITL interaction
	- An output directory is automatically created, with a name demo_acquisition_YY-MM-DD-seed
	- If ```simulated_human=False```, continue with steps 3-6 to complete the experiment.
3. To use the GUI, upload the query_it1.csv file in MolWall (saved to output_dir/loop0/query/)
	- In later iterations, refresh the browser to upload next file to MolWall
4. Rate the molecules on the scale 1-5:
	1 = not at all drd2 active (will read as 0 in the model), 5 = most certainly active (will read as 1 in the model)
	You may leave some molecules unscored.
5. Download the file to output_dir/loop0/query/ (the file will be named "query_it1_with_ratings.csv", do not modify it)
6. Press enter in the notebook to continue the script; a file query_it2.csv will be created, and so on
7. Steps 3-6 will continue 10 times, then the rest of the notebook will run to plot and save the results

In phase 2, for evaluating Chemist's component performance as Reinvent scoring component, two bash scripts have been created in output_dir/loop0/
- runs.sh
	* Script to launch Reinvent reinforcement learning runs; For each iteration X, the Chemist's component model has been saved and config_tX.json determines the corresponding Reinvent configuration. We recommend using a computation cluster and slurm: then the jobs can be submitted by

			$ sbatch runs.sh

- run_sampling.sh
	* Evaluates the resulting Reinvent agent by sampling molecules from it
	* Once the jobs from runs.sh have completed, run using

			$ sbatch run_sampling.sh

Collect, analyze and plot the results using a notebook 'evaluate_results_Task2.ipynb' (uses cc_env_reinvent)
- Use this notebook to plot the average oracle scores at each iteration after running Reinvent configurations determined in runs.sh and run_sampling.sh



How to cite
-------------------

Sundin, I., Voronov, A., Xiao, H. et al. Human-in-the-loop assisted de novo molecular design. J Cheminform 14, 86 (2022). https://doi.org/10.1186/s13321-022-00667-8
