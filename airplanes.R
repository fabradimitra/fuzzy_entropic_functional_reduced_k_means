require(clue)
require(mclust)
source("perm_hungarian_fast.R")
source("kspline.R")
source("randgenuc.R")
source("loss_function.R")
source("FERFRKM.R")
source("CV_FERFRKM.R")
source("init_FERFRKM.R")
source("make_folds.R")
source("rand_orthogonal.R")
source("preggq_int.R")
# Hyper-parameters
randomstarts <- 5
seed <- 123
kmeans_starts <- 20
#
# load data
load("data/Plane.RData")
X <- scale(X, center = TRUE, scale = FALSE)
res_kspline <- kspline(seq_len(ncol(X)))
K <- res_kspline$K
Pk <- res_kspline$Pk
Lk <- res_kspline$Lk
I <- nrow(X)
J <- ncol(X)
# load model selection
load("data/modelsel_plane_XB.RData")
best_idx <- which.min(modelsel$XB)
G <- modelsel$G[best_idx]
Q <- modelsel$Q[best_idx]
# Refit best model
cur_loss <- Inf
for (start in seq_len(randomstarts)) {
  if (start == 1) {
    init <- init_FERFRKM(X, G, Q, seed = seed, nstart_kmeans = kmeans_starts)
  } else {
    U_init <- randgenuc(I, G)
    A_init <- rand_orthogonal(G, Q)
    B_init <- t(t(A_init) %*% solve(t(U_init) %*% U_init) %*% t(U_init) %*% X)
    init <- list(U = U_init, A = A_init, B = B_init)
  }
  # Run FERFRKM algorithm
  res_cur <- FERFRKM(
    C = X,
    K = K,
    Pk = Pk,
    Lk = Lk,
    U = init$U,
    A = init$A,
    B = init$B,
    lambda = modelsel$lambda[best_idx],
    gamma = modelsel$gamma[best_idx],
    max_iter = Inf,
    tol = 1e-8
    )
  if (cur_loss > res_cur$loss_function) {
    res <- res_cur
    cur_loss <- res$loss_function
  }
}
pred_class <- max.col(res$U, ties.method = "first")
conf_mat_raw <- table(True = tcm, Predicted = pred_class)

plot_confusion_matrix <- function(mat, main) {
  mat <- as.matrix(mat)
  nr <- nrow(mat)
  nc <- ncol(mat)
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  par(mar = c(5, 5, 4, 2) + 0.1)
  plot.new()
  plot.window(
    xlim = c(0.5, nc + 0.5),
    ylim = c(nr + 0.5, 0.5),
    xaxs = "i",
    yaxs = "i"
  )
  title(ylab = "True")
  #mtext(main, side = 1, line = 3)
  mtext("Predicted", side = 3, line = 3)
  abline(v = seq(0.5, nc + 0.5, by = 1), h = seq(0.5, nr + 0.5, by = 1), col = "grey85")
  box()
  axis(3, at = seq_len(nc), labels = colnames(mat))
  axis(2, at = seq_len(nr), labels = rownames(mat), las = 1)
  for (i in seq_len(nr)) {
    for (j in seq_len(nc)) {
      text(j, i, labels = mat[i, j])
    }
  }
}
par(mfrow=c(1,1))
plot_confusion_matrix(conf_mat_raw, "Confusion matrix")
cluster_labels_est <- max.col(res$U, ties.method = "first")
adjustedRandIndex(tcm,cluster_labels_est)
est_centroids <- res$A %*% t(res$B)
centroid_1 <- t(as.matrix(colMeans(X[which(tcm == 1),])))
centroid_2 <- t(as.matrix(colMeans(X[which(tcm == 2),])))
centroid_3 <- t(as.matrix(colMeans(X[which(tcm == 3),])))
centroid_4 <- t(as.matrix(colMeans(X[which(tcm == 4),])))
centroid_5 <- t(as.matrix(colMeans(X[which(tcm == 5),])))
centroid_6 <- t(as.matrix(colMeans(X[which(tcm == 6),])))
centroid_7 <- t(as.matrix(colMeans(X[which(tcm == 7),])))
true_centroids <- rbind(centroid_1,centroid_2,centroid_3,centroid_4,centroid_5,centroid_6,centroid_7)
# Plot the centroids and their reconstruction for one iteration ----
tt <- seq(1, 144, length.out = 400)
t_grid <- c(1:J)
# Plot all curves
Y <- apply(X, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matplot(
  tt, Y, type = "l", lwd = 1, lty = 1,
  col = c("black"),
  xlab = "", ylab = "",
  main = "Airplanes dataset"
)
# Plot curves of Mirage and Eurofighter classes:
par(mfrow=c(1,2))
Mirage <- X[which(tcm == 1),]
Ymirage <- apply(Mirage, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matplot(
  tt, Ymirage, type = "l", lwd = 1, lty = 1,
  col = c("red"),
  xlab = "", ylab = "", ylim = c(-2,2),
  main = "Mirages & Eurofighters"
)
Eurofighters <- X[which(tcm == 2),]
Yeurofighter <- apply(Eurofighters, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, Yeurofighter, lwd = 1, lty = 1,
  col = c("blue")
)
est_centroid_12 <- t(as.matrix(est_centroids[2,]))
Ymr <- apply(est_centroid_12, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, Ymr, lwd = 3, lty = 2,
  col = c("black")
)
legend("bottom", legend = c("Mirage","Eurofighter","Est. centroid"),
       col = c("red","blue","black"), lwd = 2, bty = "n")
# Centroids and estimated together: 
centroid_mir <- apply(centroid_1, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matplot(
  tt, centroid_mir, type = "l", lwd = 1, lty = 1,
  col = c("red"),
  xlab = "", ylab = "", ylim = c(-1.5,2),
  main = "Centroids"
)
centroid_euro <- apply(centroid_2, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, centroid_euro, lwd = 1, lty = 1,
  col = c("blue")
)
matlines(
  tt, Ymr, lwd = 1, lty = 1,
  col = c("black")
)
legend("bottomright", legend = c("T. Mirage","T. Eurofighter","Est. centroid"),
       col = c("red","blue","black"), lwd = 2, bty = "n")
# Centroid 3 ----
Ym <- apply(centroid_3, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matplot(
  tt, Ym, type = "l", lwd = 2, lty = 1,
  col = c("gold"),
  xlab = "", ylab = "", ylim = c(-2,2),
  main = "F14 wing closed"
)
est_centroid_3 <- t(as.matrix(est_centroids[6,]))
Ymr <- apply(est_centroid_3, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, Ymr, lwd = 2, lty = 2,
  col = c("black")
)
legend("bottomright", legend = c("T. F14 w. c.","Est. centroid"),
       col = c("gold","black"), lwd = 2, bty = "n")
# Centroid 4 ----
Ym <- apply(centroid_4, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matplot(
  tt, Ym, type = "l", lwd = 2, lty = 1,
  col = c("darkgreen"),
  xlab = "", ylab = "", ylim = c(-2,2),
  main = "F14 wing open"
)
est_centroid_4 <- t(as.matrix(est_centroids[1,]))
Ymr <- apply(est_centroid_4, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, Ymr, lwd = 2, lty = 2,
  col = c("black")
)
legend("bottomright", legend = c("T. F14 w. o.","Est. centroid"),
       col = c("darkgreen","black"), lwd = 2, bty = "n")
# Centroid 5 ----
# plot harrier curves:
par(mfrow = c(1,2))
harriers_1 <- X[which((tcm == 5)&(pred_class==5)),]
ycurvesharriers_1 <- apply(harriers_1, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matplot(
  tt, ycurvesharriers_1, type = "l", lwd = 1, lty = 1,
  col = c("blue"),
  xlab = "", ylab = "", ylim = c(-3,3),
  main = "Harriers T."
)
harriers_2 <- X[which((tcm == 5)&!(pred_class==5)),]
ycurvesharriers_2 <- apply(harriers_2, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, ycurvesharriers_2, lwd = 1, lty = 1,
  col = c("red")
)
centroid_harriers <- t(as.matrix(colMeans(X[which((tcm == 5)),])))
Yharriers <- apply(centroid_harriers, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, Yharriers, lwd = 2, lty = 2,
  col = c("black")
)
legend("bottom", legend = c("Harriers 1","Harriers 2","T. centroid"),
       col = c("blue","red","black"), lwd = 2, bty = "n")
harriers_1 <- X[which((tcm == 5)&(pred_class==5)),]
ycurvesharriers_1 <- apply(harriers_1, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matplot(
  tt, ycurvesharriers_1, type = "l", lwd = 1, lty = 1,
  col = c("blue"),
  xlab = "", ylab = "", ylim = c(-3,3),
  main = "Harriers Est."
)
harriers_2 <- X[which((tcm == 5)&!(pred_class==5)),]
ycurvesharriers_2 <- apply(harriers_2, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, ycurvesharriers_2, lwd = 1, lty = 1,
  col = c("red")
)
est_centroid_harriers1 <- t(as.matrix(est_centroids[5,]))
Ymr <- apply(est_centroid_harriers1, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, Ymr, lwd = 2, lty = 2,
  col = c("black")
)
est_centroid_harriers2 <- t(as.matrix(est_centroids[3,]))
Ymr <- apply(est_centroid_harriers2, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, Ymr, lwd = 2, lty = 2,
  col = c("black")
)
legend("bottom", legend = c("Harriers 1","Harriers 2","Est. centroids"),
       col = c("blue","red","black"), lwd = 2, bty = "n")
# Plot curves of F22 and F15 classes:
par(mfrow = c(1,2))
F22 <- X[which(tcm == 6),]
YF22 <- apply(F22, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matplot(
  tt, YF22, type = "l", lwd = 1, lty = 1,
  col = c("steelblue2"),
  xlab = "", ylab = "", ylim = c(-2.5,1.5),
  main = "F-22 & F-15"
)
F15 <- X[which(tcm == 7),]
YF15 <- apply(F15, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, YF15, lwd = 1, lty = 1,
  col = c("hotpink3")
)
est_centroid_F22_15 <- t(as.matrix(est_centroids[4,]))
Ymr <- apply(est_centroid_F22_15, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, Ymr, lwd = 2, lty = 2,
  col = c("black")
)
legend("bottom", legend = c("F-22","F-14","Est. Centroid"),
       col = c("steelblue2","hotpink3","black"), lwd = 2, bty = "n")
# Centroids and estimated together: 
centroid_F22 <- apply(centroid_6, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matplot(
  tt, centroid_F22, type = "l", lwd = 1, lty = 1,
  col = c("steelblue2"),
  xlab = "", ylab = "", ylim = c(-2.5,1.5),
  main = "Centroids"
)
centroid_F15 <- apply(centroid_7, 1, function(y) splinefun(t_grid, y, method = "natural")(tt))
matlines(
  tt, centroid_F15, lwd = 1, lty = 1,
  col = c("hotpink3")
)
matlines(
  tt, Ymr, lwd = 1, lty = 1,
  col = c("black")
)
legend("bottom", legend = c("T. F-22","T. F-14","Est. Centroid"),
       col = c("steelblue2","hotpink3","black"), lwd = 2, bty = "n")

