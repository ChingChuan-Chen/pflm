library(fdapace)

bench_lwls1d <- function(bw, kernel, win, xin, xout) {
    gc()
    yin <- xin ** 2 - xin * 3 + 1 + rnorm(length(xin), 0, 0.25)
    st <- proc.time()
    fdapace:::Lwls1D(bw, kernel, win, xin, yin, xout)
    et <- proc.time()
    return((et - st)[3])
}

n <- 1e4
bw <- 0.05
xin <- seq(0, 1, length.out=n)
win <- rep(1, n)
xout <- seq(0, 1, length.out=n*2)

print(sessionInfo())
kernel_types <- c("gauss", "epan")
num_replications <- 30
run_times <- replicate(length(kernel_types), numeric(num_replications), simplify = FALSE)
names(run_times) <- kernel_types
for (kernel in kernel_types) {
    run_times[[kernel]] <- numeric(num_replications)
    for (i in seq_len(num_replications)) {
        set.seed(42)
        run_times[[kernel]][i] <- bench_lwls1d(bw, kernel, win, xin, xout)
    }
}

for (kernel in kernel_types) {
    run_times_remove <- sort(run_times[[kernel]])[2:(num_replications-1)]
    cat(sprintf("Average time (remove fastest and slowest) for %d replications with sample size %d on lwls1d with %s kernel runs: %.6f seconds\n",
                num_replications, n, kernel, mean(run_times_remove)))
    cat(sprintf("Standard deviation of run times: %.6f seconds\n", sd(run_times_remove)))
}
print(run_times)

# R version 4.5.1 (2025-06-13 ucrt)
# Platform: x86_64-w64-mingw32/x64
# Running under: Windows 11 x64 (build 26100)

# Matrix products: default
#   LAPACK version 3.12.1

# locale:
# [1] LC_COLLATE=Chinese (Traditional)_Taiwan.utf8  LC_CTYPE=Chinese (Traditional)_Taiwan.utf8
# [3] LC_MONETARY=Chinese (Traditional)_Taiwan.utf8 LC_NUMERIC=C
# [5] LC_TIME=Chinese (Traditional)_Taiwan.utf8

# time zone: Asia/Taipei
# tzcode source: internal

# attached base packages:
# [1] stats     graphics  grDevices utils     datasets  methods   base

# other attached packages:
# [1] fdapace_0.6.0

# loaded via a namespace (and not attached):
#  [1] Matrix_1.7-3        gtable_0.3.6        compiler_4.5.1      rpart_4.1.24        Rcpp_1.1.0          htmlTable_2.4.3
#  [7] stringr_1.5.1       gridExtra_2.3       cluster_2.1.8.1     Hmisc_5.2-3         scales_1.4.0        fastmap_1.2.0
# [13] lattice_0.22-7      ggplot2_3.5.2       R6_2.6.1            Formula_1.2-5       knitr_1.50          MASS_7.3-65
# [19] htmlwidgets_1.6.4   backports_1.5.0     checkmate_2.3.2     tibble_3.3.0        nnet_7.3-20         pillar_1.11.0
# [25] RColorBrewer_1.1-3  rlang_1.1.6         stringi_1.8.7       xfun_0.52           cli_3.6.5           magrittr_2.0.3
# [31] digest_0.6.37       grid_4.5.1          rstudioapi_0.17.1   base64enc_0.1-3     lifecycle_1.0.4     vctrs_0.6.5
# [37] pracma_2.4.4        data.table_1.17.8   evaluate_1.0.4      glue_1.8.0          numDeriv_2016.8-1.1 farver_2.1.2
# [43] colorspace_2.1-1    rmarkdown_2.29      foreign_0.8-90      tools_4.5.1         pkgconfig_2.0.3     htmltools_0.5.8.1

# Average time (remove fastest and slowest) for 30 replications with sample size 10000 on lwls1d with gauss kernel runs: 8.143571 seconds
# Standard deviation of run times: 0.324165 seconds
# Average time (remove fastest and slowest) for 30 replications with sample size 10000 on lwls1d with epan kernel runs: 0.236429 seconds
# Standard deviation of run times: 0.010616 seconds
