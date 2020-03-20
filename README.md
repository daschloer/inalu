# iNALU: Improved Neural Arithmetic Logic Unit
This repository contains the code for the paper [iNALU: Improved Neural Arithmetic Logic Unit](https://arxiv.org/abs/2003.07629) for Tensorflow 1.8.

The python scripts are used as follows:
* [nalu_architectures.py](nalu_architectures.py): contains the code for the iNALU and NALU cells as well as the two layer and deep NALU architecture
	* nalu_paper_matrix_layer: Code for the original NALU (Task et al. 2018), implemented with matrix gating
	* nalu_paper_vector_layer: Code for the original NALU (Task et al. 2018), implemented with vector gating
	* naluv_layer: Code for the iNALU with our improvements, implemented with vector gating
	* nalum_layer: Code for the iNALU with our improvements, implemented with matrix gating
	* nalui1_layer: Code for the iNALU with our improvements and independent gating with shared weights between add and mul
	* nalui2_layer: Code for the iNALU with our improvements and independent gating with seperate weights for add and mul

* [nalu_syn_simple_arith.py](nalu_syn_simple_arith.py): Training Code for experiments 1 and 2
* [nalu_syn_simple_func.py](nalu_syn_simple_func.py): Training Code for experiments 3 and 4
* [results.py](results.py): Code to reproduce the result plots from result files
* [runner.py](runner.py): Script to run experiments (parallel) for different parameters synced over a shared filesystem. The parameter config files are created by experimentplanner.py (see subfolder experiment_runs/exp*)
* [requirements.txt](runner.py): pip requirements file

The folder [experiment_runs](experiment_runs) contains result data and plots for each experiment mentioned in the paper. The file experimentplanner.py in each subfolder defines the parameter configurations used for each experiment, which can be executed by runner.py.
The experimentplanner.py scripts also support checking out a designated git repository with the code (with a specific revision/tag) for mass-replication.

If you find this code useful for your own research, please cite: https://arxiv.org/abs/2003.07629

	    @article{Schlor2020iNALUIN,
	      title={iNALU: Improved Neural Arithmetic Logic Unit},
	      author={Daniel Schlör and Markus Ring and Andreas Hotho},
	      journal={arXiv preprint arXiv:2003.07629},
	      year={2020}
	    }
