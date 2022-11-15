using JLD
using LinearAlgebra


function retrieve_results()
	#c1d1 = load("casecode1/divfactor1/results.jld")
	mat_results_cpu = zeros(2,6)
	mat_results_gpu = zeros(2,6)
	case_counter = 1
	divfactor_counter = 1
	for caseind=[1 2]
		divfactor_counter = 1
		for divfactor=[1 2 4 8 12 16]
			dhere = load(string("casecode",caseind,"/divfactor",divfactor,"/results.jld"))
			cpu_time = dhere["final_time_cpu"]
			gpu_time = dhere["final_time_cuda"]
			mat_results_cpu[case_counter,divfactor_counter]= cpu_time
			mat_results_gpu[case_counter,divfactor_counter]= gpu_time
			divfactor_counter = divfactor_counter + 1
		end
		case_counter = case_counter + 1
	end
	return mat_results_cpu, mat_results_gpu
end

(mat_results_cpu,mat_results_gpu) = retrieve_results()

save("total_results.jld","mat_results_cpu",mat_results_cpu,"mat_results_gpu",mat_results_gpu)


