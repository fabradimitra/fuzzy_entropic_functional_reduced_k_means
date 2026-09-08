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
randomstarts <- 1
randomstarts_cv <- 1
kmeans_starts <- 20
lambda_init <- 1
gamma_init <- 8
sig <- 4
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
  cos(10 * t)
}
psi2_wiggly <- function(t) {
  sin(10 * t)
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
# Monte Carlo simulations
nsim <- 100
Res<-matrix(0,nrow = nsim, ncol = 4)
for(iter in c(1:nsim)){
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
    # if(start == 1){
    #   init <- init_FERFRKM(X, G, Q, seed = iter, nstart_kmeans = kmeans_starts) 
    # }else{
    #   U_init <- randgenuc(I, G)
    #   A_init <- matrix(rnorm(G*Q),G,Q)
    #   B_init <- t(t(A_init)%*%solve(t(U_init)%*%U_init)%*%t(U_init)%*%X)
    #   init <- list(U=U_init, A=A_init, B=B_init)
    # }
    U_init <- randgenuc(I,G)
    Cbar_init <- diag(1/colSums(U_init)) %*% t(U_init) %*% X
    sv <- svd(Cbar_init, nu = Q, nv = Q)
    A_init <- sv$u[, 1:Q, drop = FALSE]
    B_init <- t(t(A_init)%*%solve(t(U_init)%*%U_init)%*%t(U_init)%*%X)
    init <- list(U=U_init, A=A_init, B=B_init)
    # Run FERFRKM algorithm
    res_g0 <-  FERFRKM(C=X,
                        K=K,
                        Pk=Pk,
                        Lk=Lk,
                        U=init$U,
                        A=init$A,
                        B=init$B,
                        lambda= lambda_best,
                        gamma = 0,
                        max_iter = Inf,
                        tol = 1e-6)
    ABp <- res_g0$A %*% t(res_g0$B)
    perm <- perm_hungarian_fast(Curves, ABp, J)
    cluster_labels_est <- max.col(res_g0$U, ties.method = "first")
    ARI_km <- adjustedRandIndex(cluster_labels,cluster_labels_est)
    SqE_km <- sum((ABp[perm,] - Curves)^2)
    res <-   FERFRKM(C=X,
                     K=K,
                     Pk=Pk,
                     Lk=Lk,
                     U=init$U,
                     A=init$A,
                     B=init$B,
                     lambda= lambda_best,
                     gamma = gamma_best,
                     max_iter = Inf,
                     tol = 1e-6)
    ABp <- res$A %*% t(res$B)
    perm <- perm_hungarian_fast(Curves, ABp, J)
    cluster_labels_est <- max.col(res$U, ties.method = "first")
    ARI_fkm <- adjustedRandIndex(cluster_labels,cluster_labels_est)
    SqE_fkm <- sum((ABp[perm,] - Curves)^2)
  }
  Res[iter,] <- c(SqE_km, ARI_km, SqE_fkm, ARI_fkm)
  cat("End Monte Carlo Simulation: ", iter, 
      " SqE KM: ", SqE_km, " ARI KM ", ARI_km, 
      " SqE FKM: ", SqE_fkm, " ARI FKM ", ARI_fkm,"\n")
}
colMeans(Res)
# Plot the centroids and their reconstruction for one iteration ----
tt <- seq(min(t_grid), max(t_grid), length.out = 400)
Ym <- apply(Curves, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matplot(
  tt, Ym, type = "l", lwd = 1, lty = 1,
  col = c("red","blue","darkgreen","orange"),
  xlab = "", ylab = ""
)
Ymr <- apply(ABp[perm,], 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
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

