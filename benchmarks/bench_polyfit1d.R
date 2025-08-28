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
