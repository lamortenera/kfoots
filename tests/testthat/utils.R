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

runs <- function(expr){
    res <- try(force(expr), TRUE)
    no_error <- !inherits(res, "try-error")
    if (no_error) {
        return(expectation(TRUE, "code generated an error", 
        "code did not generate an error"))
    }
    else {
        expectation(FALSE, paste0("threw an error:\n", res[1]), "no error thrown")
    }
}
expect_runs <- function(object, info = NULL, label = NULL){
    if (is.null(label)) {
        label <- testthat:::find_expr("object")
    }
    expect_that(object, runs, info = info, label = label)
}
