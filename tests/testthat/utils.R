modelsAreOk <- function(models, k, nrow, nbtype){
	if (length(models) != k) return(F)
	tryCatch({
		for (model in models){
			if (length(model$ps) != nrow || !all.equal(sum(model$ps),1)) return(F)
			if (!(all(is.finite(c(model$mu, model$ps))) && (unlist(model) >= 0))) return(F)
			if (nbtype=="dep" && model$r != models[[1]]$r) return(F)
			if (nbtype=="pois" && model$r != Inf) return(F)
		}
	},
	error=function(cond){
		print(cond)
		return(F)
	})
	return(T)
}
