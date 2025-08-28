library(fdapace)

bench_lwls2d <- function(bw1, bw2, kernel, w, x_grid, xout) {
  gc()
  y <- sin(pi * x_grid[,1]) * cos(pi * x_grid[,2]) + 0.1 * rnorm(nrow(x_grid))
  st <- proc.time()
  fdapace:::Lwls2D(c(bw1, bw2), kernel, x_grid, y, w, xout, xout, crosscov = FALSE)
  et <- proc.time()
  return((et - st)[3])
}

n <- 100
bw <- 0.05
x1 <- seq(0, 1, length.out=n)
x_grid <- as.matrix(expand.grid(x1, x1))
win <- rep(1, nrow(x_grid))
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
    run_times[[kernel]][i] <- bench_lwls2d(bw, bw, kernel, win, x_grid, xout)
  }
}

for (kernel in kernel_types) {
  run_times_remove <- sort(run_times[[kernel]])[2:(num_replications-1)]
  cat(sprintf("Average time (remove fastest and slowest) for %d replications with sample size %d on lwls1d with %s kernel runs: %.6f seconds\n",
              num_replications, n, kernel, mean(run_times_remove)))
  cat(sprintf("Standard deviation of run times: %.6f seconds\n", sd(run_times_remove)))
}
print(run_times)
