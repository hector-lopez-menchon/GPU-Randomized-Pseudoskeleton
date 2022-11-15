#!/bin/bash


for i in {1,2,3,4};
	do
		cp -r seed casecode"$i"
		cd casecode"$i"
		echo -e "$i" > parameters.txt
		cd ../
	done


for i in {1,2,3,4};
	do
		#mkdir casecode"$i"
		cd casecode"$i"
		module load cuda/11.0
		srun -p csl  --gres=gpu:1  julia  compres_single_block_spheres_for_lambda_sweep.jl > compres_single_block_spheres_for_lambda_sweep.txt  &
		cd ../
	done


