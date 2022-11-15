#!/bin/bash

####Create folders with problem instances
for i in {1,2};
	do
		mkdir casecode"$i"
		cd casecode"$i"
		for j in {1,2,4,8,12,16}
			do
				cp -r ../seed divfactor"$j"
				cd divfactor"$j"
				echo -e "$i\n$j" > parameters.txt
				cd ../
			done
		cd ../
	done

#Execute them
for i in {1,2};
        do
                cd casecode"$i"
                for j in {1,2,4,8,12,16}
                        do
                                cd divfactor"$j"
                                module load cuda/11.0
                                srun -p csl  --gres=gpu:1  /home/usuaris/csl/hector.lopez.m/Documentos/julia_trying/julia/julia  compres_single_block_spheres_for_sweep2.jl > compres_single_block_spheres_for_sweep2.txt  &
                                cd ../
                        done
                cd ../
        done


