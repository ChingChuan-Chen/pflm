suppressMessages(library(fdapace))

args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args) >= 1) as.integer(args[1]) else as.integer(1e4)
num_replications <- if (length(args) >= 2) as.integer(args[2]) else 30
bw <- if (length(args) >= 3) as.numeric(args[3]) else 0.05

bench_lwls1d <- function(bw, kernel, win, xin, xout) {
    gc()
    yin <- xin^2 - xin * 3 + 1 + rnorm(length(xin), 0, 0.25)
    st <- proc.time()
    out <- fdapace:::Lwls1D(bw, kernel, win, xin, yin, xout)
    et <- proc.time()
    rm(out)
    (et - st)[3]
}

summarize_times <- function(label, times_s) {
    if (length(times_s) > 2) {
        times_trimmed <- sort(times_s)[2:(length(times_s) - 1)]
    } else {
        times_trimmed <- times_s
    }
    std_s <- if (length(times_trimmed) > 1) stats::sd(times_trimmed) else 0.0
    cat(sprintf(
        "Average time (remove fastest and slowest) for %d replications on %s: %.6f seconds\n",
        length(times_s), label, mean(times_trimmed)
    ))
    cat(sprintf("Standard deviation of run times: %.6f seconds\n", std_s))
}

xin <- seq(0, 1, length.out = n)
win <- rep(1, n)
xout <- seq(0, 1, length.out = n * 2)

cat(sprintf("R version: %s\n", R.version.string))
print(sessionInfo())
cat(sprintf("n=%d, num_replications=%d, bandwidth=%.4f\n", n, num_replications, bw))

kernel_types <- c("gauss", "epan")
run_times <- setNames(replicate(length(kernel_types), numeric(num_replications), simplify = FALSE), kernel_types)
for (kernel in kernel_types) {
    for (i in seq_len(num_replications)) {
        set.seed(42)
        run_times[[kernel]][i] <- bench_lwls1d(bw, kernel, win, xin, xout)
    }
}

summarize_times("Lwls1D (gauss)", run_times[["gauss"]])
summarize_times("Lwls1D (epan)", run_times[["epan"]])

print(run_times)
