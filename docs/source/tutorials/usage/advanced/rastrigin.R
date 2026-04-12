rastrigin <- function(x, A=10) {
  n <- length(x)
  A*n + sum(x^2 - A*cos(2*pi*x))
}

args <- commandArgs(trailingOnly=TRUE)
file_name <- args[1]
x <- as.numeric(args[2:4])

out_path <- file.path("data", file_name)
fx <- rastrigin(x)
write(fx, file=out_path)
