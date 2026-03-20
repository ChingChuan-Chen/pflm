suppressMessages(library(fdapace))

args <- commandArgs(trailingOnly = TRUE)
n_side <- if (length(args) >= 1) as.integer(args[1]) else 100L
num_replications <- if (length(args) >= 2) as.integer(args[2]) else 30
bw1 <- if (length(args) >= 3) as.numeric(args[3]) else 0.05
bw2 <- if (length(args) >= 4) as.numeric(args[4]) else 0.05

bench_lwls2d <- function(bw1, bw2, kernel, w, x_grid, xout) {
  gc()
  y <- sin(pi * x_grid[, 1]) * cos(pi * x_grid[, 2]) + 0.1 * rnorm(nrow(x_grid))
  st <- proc.time()
  out <- fdapace:::Lwls2D(c(bw1, bw2), kernel, x_grid, y, w, xout, xout, crosscov = FALSE)
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

x1 <- seq(0, 1, length.out = n_side)
x_grid <- as.matrix(expand.grid(x1, x1))
win <- rep(1, nrow(x_grid))
xout <- seq(0, 1, length.out = n_side * 2)

cat(sprintf("R version: %s\n", R.version.string))
print(sessionInfo())
cat(sprintf(
  "n_side=%d, num_replications=%d, bandwidth1=%.4f, bandwidth2=%.4f\n",
  n_side, num_replications, bw1, bw2
))

kernel_types <- c("gauss", "epan")
run_times <- setNames(replicate(length(kernel_types), numeric(num_replications), simplify = FALSE), kernel_types)
for (kernel in kernel_types) {
  for (i in seq_len(num_replications)) {
    set.seed(42)
    run_times[[kernel]][i] <- bench_lwls2d(bw1, bw2, kernel, win, x_grid, xout)
  }
}

summarize_times("Lwls2D (gauss)", run_times[["gauss"]])
summarize_times("Lwls2D (epan)", run_times[["epan"]])

print(run_times)
