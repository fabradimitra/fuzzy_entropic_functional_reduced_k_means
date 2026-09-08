CV_FERFRKM <- function(
  Xcv,
  G,
  Q,
  K = kspline(seq_len(ncol(Xcv)))$K,
  Pk = NULL,
  Lk = NULL,
  lambda_init = 1,
  gamma_init = 1,
  folds = 10,
  fold_ids = NULL, 
  max_iter = Inf,
  tol = 1e-8,
  nstart_kmeans = 10,
  seed = 123,
  randomstarts = 5
) {
  stopifnot(is.matrix(Xcv), G >= 2, Q >= 1, Q <= min(G, ncol(Xcv)))
  #
  if (is.null(Pk) || is.null(Lk)) {
    resK <- kspline(seq_len(ncol(Xcv)))
    Pk <- resK$Pk
    Lk <- resK$Lk
  }
  #
  if (is.null(fold_ids)) {
    fold_ids <- make_folds(nrow(Xcv), k = folds, seed = seed)
  }
  folds <- sort(unique(fold_ids)) # fold labels
  #
  fun_optim <- function(vars, folds, fold_ids, G, Q, 
    X, K, Pk, Lk, randomstarts, nstart_kmeans, seed, max_iter, tol){
    fold_scores <- numeric(length(folds))
    for (fold_idx in seq_along(folds)) {
      fold <- folds[fold_idx]
      train_idx <- fold_ids != fold
      valid_idx <- !train_idx
      X_train <- X[train_idx, , drop = FALSE]
      X_valid <- X[valid_idx, , drop = FALSE]
      score_fin <- Inf
      for(start in seq_len(randomstarts)){
        # if(start == 1){
        # init <- init_FERFRKM(X = X_train, G = G, Q = Q, seed = seed, nstart_kmeans = nstart_kmeans)
        # }else{
        # U_init <- randgenuc(nrow(X_train), G)
        # A_init <- matrix(rnorm(G*Q),G,Q)
        # B_init <- t(t(A_init) %*% solve(t(U_init) %*% U_init) %*% t(U_init) %*% X_train)
        # init <- list(U = U_init, A = A_init, B = B_init)
        # }
        U_init <- randgenuc(nrow(X_train),G)
        Cbar_init <- diag(1/colSums(U_init)) %*% t(U_init) %*% X_train
        sv <- svd(Cbar_init, nu = Q, nv = Q)
        A_init <- sv$u[, 1:Q, drop = FALSE]
        B_init <- t(t(A_init)%*%solve(t(U_init)%*%U_init)%*%t(U_init)%*%X_train)
        init <- list(U=U_init, A=A_init, B=B_init)
        fit <- tryCatch(
            FERFRKM(
              C = X_train,
              K = K,
              Pk = Pk,
              Lk = Lk,
              U = init$U,
              A = init$A,
              B = init$B,
              lambda = vars[1],
              gamma = vars[2],
              max_iter = max_iter,
              tol = tol
            ),
          error = function(e) NULL
        )
        if (is.null(fit)) {
            next
        }
        Centers <- fit$A %*% t(fit$B)   # G x J
        xnorm2_valid <- rowSums(X_valid^2)     # length n_valid
        cnorm2 <- rowSums(Centers^2)           # length G
        Dist2_valid <- outer(xnorm2_valid, cnorm2, "+") - 2 * (X_valid %*% t(Centers))
        if(vars[2]==0){
          U_valid <- matrix(0,nrow = nrow(X_valid), ncol = G)
          U_valid[cbind(nrow(X_valid), max.col(-Dist2_valid, ties.method = "first"))] <- 1
        }else{
          Dist2_valid <- t(apply(Dist2_valid,1,function(x){x<-x-min(x)}))
          U_valid <- exp(-Dist2_valid / vars[2])
          U_valid <- U_valid / matrix(rowSums(U_valid),nrow(X_valid),G)
        }
        D_valid <- diag(sqrt(colSums(U_valid)))
        D2_valid <- D_valid^2
        Cbar_valid <- diag(1/diag(D2_valid)) %*% t(U_valid) %*% X_valid
        cur_score <- loss_function(U_valid, X_valid, Cbar_valid, D_valid,
          fit$A, fit$B, K, vars[1], vars[2])$wdev
        if (is.null(cur_score)|is.na(cur_score)) {
            next
        }
        if(cur_score<score_fin){
          score_fin <- cur_score
        }
      }
      fold_scores[fold_idx] <- score_fin
      cat("Fold ", fold_idx, "Score ", score_fin)
    }
    mean(fold_scores)
  }
  optim(
    par = c(lambda_init,gamma_init),
    fn = fun_optim,
    method = "Nelder-Mead",
    folds = folds, 
    fold_ids = fold_ids,
    G = G,
    Q = Q,
    X = Xcv,
    K = K,
    Pk = Pk,
    Lk = Lk,
    randomstarts = randomstarts,
    nstart_kmeans = nstart_kmeans,
    seed = seed,
    max_iter = max_iter,
    control = list(maxit = 100),
    tol = tol
    )
}
    
