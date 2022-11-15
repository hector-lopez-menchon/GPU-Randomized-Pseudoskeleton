using JLD
using LinearAlgebra

function retrieve_results(nlc,n_ors)
	#nlc: number of lambda cases
	#n_ors: number of compressions of different size
	
	errCUR = zeros(n_ors,nlc)
	errACA = zeros(n_ors,nlc)

	timeCUR = zeros(n_ors,nlc)
	timeACA = zeros(n_ors,nlc)
	
	nelsCUR = zeros(n_ors,nlc)
	nelsACA = zeros(n_ors,nlc)

	for ii=1:nlc
		dhere = load(string("casecode",ii,"/results_sweep_compression.jld"))
		errvecCUR = dhere["errvecCUR"]
		errvecACA = dhere["errvecACA"]
		timevecCUR = dhere["timevecCUR"]
		timevecACA = dhere["timevecACA"]
		nels_vecCUR = dhere["nels_vecCUR"]
		nels_vecACA = dhere["nels_vecACA"]
		
		errCUR[:,ii] = errvecCUR
		errACA[:,ii] = errvecACA

		timeCUR[:,ii] = timevecCUR
		timeACA[:,ii] = timevecACA

		nelsCUR[:,ii] = nels_vecCUR
		nelsACA[:,ii] = nels_vecACA
	end

	return errCUR, errACA, timeCUR, timeACA, nelsCUR, nelsACA
end

dsample = load(string("casecode",1,"/results_sweep_compression.jld"))
errvecCURsample = dsample["errvecCUR"]
n_ors = length(errvecCURsample)

(errCUR,errACA,timeCUR,timeACA,nelsCUR,nelsACA) = retrieve_results(4,n_ors)

save("global_results.jld","errCUR",errCUR,"errACA",errACA,"timeCUR",timeCUR,"timeACA",timeACA,"nelsCUR",nelsCUR,"nelsACA",nelsACA)


