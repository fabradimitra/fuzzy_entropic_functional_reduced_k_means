require(mclust)
require(clue)
require(MASS)
#
source("kspline.R")
source("randgenuc.R")
source("rand_orthogonal.R")
source("loss_function.R")
source("FERFRKM.R")
source("perm_hungarian_fast.R")
source("CV_FERFRKM.R")
source("init_FERFRKM.R")
source("make_folds.R")
# Simulation preparation -----
randomstarts <- 5
randomstarts_cv <- 1
kmeans_starts <- 20
lambda_init <-0.01
gamma_init <- 1
# Set up dimensions and centroids
J <- 101
I <- 100
Q <- 2
G <- 6
# smooth smooth
psi1_smooth <- function(t) {
  t + sin(pi * t) * exp(-t)
}
psi2_smooth <- function(t) {
  cos(3 + pi * t)
}
psi1_wiggly <- function(t) {
  sin(10 * t)
}
psi2_wiggly <- function(t) {
  cos(10 * t)
}
# True A matrix (orthogonal)
set.seed(123)
A <- matrix(rnorm(G*Q),G,Q)
# Evaluate the curves at a grid of observed points
t_grid <- seq(-1, 1, length.out = J)
f1 <- psi1_smooth(t_grid)
f2 <- psi2_wiggly(t_grid)
B <- cbind(f1,f2)
Curves <- A %*% t(B)
res <- kspline(t_grid)
K <- res$K
Pk <- res$Pk
Lk <- res$Lk
#
IJ <- diag(J)
sig <- 12
# Monte Carlo simulations
for(iter in c(5)){
  set.seed(iter)
  # Draw data from a mixture of Gaussian distributions
  Dummy_labels <- t(rmultinom(
    n = I, size = 1, 
    prob = rep(1/G,G)
  ))
  cluster_labels <- max.col(Dummy_labels, ties.method = "first")
  X <- t(sapply(cluster_labels, function(lbl){
    mvrnorm(1, mu = Curves[lbl,], Sigma = sig*IJ)
  }
  ))
  # Cross validation
  cv_res <- CV_FERFRKM(
      Xcv = X,
      G = G,
      Q = Q,
      K = K,
      Pk = Pk,
      Lk = Lk,
      lambda_init = lambda_init,
      gamma_init = gamma_init,
      folds = 5,
      max_iter = Inf,
      tol = 1e-8,
      nstart_kmeans = kmeans_starts,
      seed = iter,
      randomstarts = randomstarts_cv
    )
  lambda_best <- cv_res$par[1]
  gamma_best <- cv_res$par[2]
  # Fit the best combination:
  cur_loss <- Inf
  for(start in seq_len(randomstarts)){
    if(start == 1){
      init <- init_FERFRKM(X, G, Q, seed = iter, nstart_kmeans = kmeans_starts) 
    }else{
      U_init <- randgenuc(I, G)
      A_init <- matrix(rnorm(G*Q),G,Q)
      B_init <- t(t(A_init)%*%solve(t(U_init)%*%U_init)%*%t(U_init)%*%X)
      init <- list(U=U_init, A=A_init, B=B_init)
    }
    # Run FERFRKM algorithm
    res_cur <-  tryCatch(
              FERFRKM(C=X,
                      K=K,
                      Pk=Pk,
                      Lk=Lk,
                      U=init$U,
                      A=init$A,
                      B=init$B,
                      lambda= lambda_best,
                      gamma = gamma_best,
                      max_iter = Inf,
                      tol = 1e-8),
      error = function(e) {
        warning(e)
        NULL
      }
    )
    if (is.null(res_cur)) {
      next
    }
    if(cur_loss>res_cur$loss_function){
      res <- res_cur
      cur_loss <- res$loss_function
    }
  }
  cluster_labels_est <- max.col(res$U, ties.method = "first")
  ARI <- adjustedRandIndex(cluster_labels,cluster_labels_est)
  ABp <- res$A %*% t(res$B)
  out <- list(
    i = iter,
    gamma_best = gamma_best,
    lambda_best = lambda_best,
    ARI = ARI,
    est_centroids = ABp
  )
  perm <- perm_hungarian_fast(Curves, out$est_centroids, J)
  mean((Dummy_labels-res$U[,perm])^2)
  cat("End Monte Carlo Simulation: ", iter, " Lambda* ", lambda_best, " Gamma* ", gamma_best,
    "ARI: ", ARI, " SqE: ", sum(ABp[perm,] - Curves)^2,"\n")
}
# Plot the centroids and their reconstruction for one iteration ----
tt <- seq(min(t_grid), max(t_grid), length.out = 400)
Ym <- apply(Curves, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matplot(
  tt, Ym, type = "l", lwd = 1, lty = 1,
  col = c("red","blue","darkgreen","orange"),
  xlab = "", ylab = ""
)
Ymr <- apply(out$est_centroids[perm,], 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, Ymr, lwd = 4, lty = 2,
  col = c("red","blue","darkgreen","orange")
)
legend(
  "bottomright",
  legend = paste0("cluster ", 1:4),
  col = c("red","blue","darkgreen","orange"),
  lwd = 2, bty = "n"
)
cols <- c("red", "blue", "darkgreen", "orange")[cluster_labels]
Y <- t(apply(X, 1, function(y) splinefun(t_grid, y, method = "natural")(tt)))

matplot(tt, t(Y), type = "l", lty = 1, col = cols, lwd = 2,
        xlab = "t", ylab = "spline value")
legend("bottomright", legend = paste("label", 1:4),
       col = c("red","blue","darkgreen","orange"), lwd = 2, bty = "n")

