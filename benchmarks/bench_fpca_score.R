suppressMessages(library(fdapace))

# Base data and model parameters from fdapace/test_scores.R
base_Ly <- list(c(1, 2, 2), c(3, 4), c(4, 5))
base_Lt <- list(c(0.1, 0.2, 0.3), c(0.2, 0.3), c(0.1, 0.3))
mu <- c(2.5, 2.5, 11 / 3)
fitted_cov <- matrix(
  c(
    7.33511649, 0.47405257, 5.63170887,
    0.47405257, 0.52865747, 0.57323255,
    5.63170887, 0.57323255, 4.41181136
  ),
  ncol = 3,
  byrow = TRUE
)
lambda <- c(0.58938792, 0.05082422)
phi <- matrix(
  c(
    3.52004852, -0.79550581,
    0.28836234, 3.07203424,
    2.72817755, 0.70169923
  ),
  ncol = 2,
  byrow = TRUE
)
obs_grid <- c(0.1, 0.2, 0.3)
sigma2 <- 0.0

args <- commandArgs(trailingOnly = TRUE)
num_subjects <- if (length(args) >= 1) as.integer(args[1]) else 3000
num_replications <- if (length(args) >= 2) as.integer(args[2]) else 30

build_dataset <- function(num_subjects) {
  idx <- ((seq_len(num_subjects) - 1L) %% length(base_Ly)) + 1L
  Ly <- lapply(idx, function(i) as.double(base_Ly[[i]]))
  Lt <- lapply(idx, function(i) as.double(base_Lt[[i]]))
  list(Ly = Ly, Lt = Lt)
}

bench_ce <- function(Ly, Lt) {
  gc()
  optns <- list(methodRho = "trunc", verbose = FALSE)
  st <- proc.time()
  out <- fdapace:::GetCEScores(Ly, Lt, optns, mu, obs_grid, fitted_cov, lambda, phi, sigma2)
  et <- proc.time()
  rm(out)
  (et - st)[3]
}

bench_in <- function(Ly, Lt) {
  gc()
  optns <- list(shrink = FALSE, verbose = FALSE)
  st <- proc.time()
  out <- mapply(
    function(yvec, tvec) fdapace:::GetINScores(yvec, tvec, optns, obs_grid, mu, lambda, phi, sigma2),
    Ly,
    Lt,
    SIMPLIFY = FALSE
  )
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

cat(sprintf("R version: %s\n", R.version.string))
print(sessionInfo())
cat(sprintf("num_subjects=%d, num_replications=%d\n", num_subjects, num_replications))

data <- build_dataset(num_subjects)
Ly <- data$Ly
Lt <- data$Lt

run_times <- list(CE = numeric(num_replications), IN = numeric(num_replications))
for (i in seq_len(num_replications)) {
  run_times$CE[i] <- bench_ce(Ly, Lt)
  run_times$IN[i] <- bench_in(Ly, Lt)
}

summarize_times("GetCEScores", run_times$CE)
summarize_times("GetINScores (mapply over subjects)", run_times$IN)

print(run_times)
