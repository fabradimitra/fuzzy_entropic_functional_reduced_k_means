CV_FERFRKM <- function(
  Xtr,
  G,
  Q,
  K = kspline(seq_len(ncol(Xtr)))$K,
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
  stopifnot(is.matrix(Xtr), G >= 2, Q >= 1, Q <= min(G, ncol(Xtr)))
  #
  if (is.null(Pk) || is.null(Lk)) {
    ks <- kspline(seq_len(ncol(Xtr)))
    Pk <- ks$Pk
    Lk <- ks$Lk
  }
  #
  if (is.null(fold_ids)) {
    fold_ids <- make_folds(nrow(Xtr), k = folds, seed = seed)
  }
  folds <- sort(unique(fold_ids)) # fold labels
  #
  cv_xie_beni_optim <- function(vars, folds, fold_ids, G, Q, 
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
        if(start == 1){
        init <- init_FERFRKM(
        X = X_train,
        G = G,
        Q = Q,
        seed = seed,
        nstart_kmeans = nstart_kmeans
        )
        }else{
        U_init <- randgenuc(nrow(X_train), G)
        A_init <- rand_orthogonal(G, Q)
        B_init <- t(t(A_init) %*% solve(t(U_init) %*% U_init) %*% t(U_init) %*% X_train)
        init <- list(U = U_init, A = A_init, B = B_init)
        }
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
        centers <- fit$A %*% t(fit$B)   # G x J
        cnorm2_valid <- rowSums(X_valid^2)     # length n_valid
        vnorm2 <- rowSums(centers^2)           # length G
        dist2_valid <- outer(cnorm2_valid, vnorm2, "+") - 2 * (X_valid %*% t(centers))
        if(vars[2]==0){
          U_valid <- matrix(0,nrow = nrow(X_valid), ncol = G)
          U_valid[cbind(nrow(X_valid), max.col(-dist2_valid, ties.method = "first"))] <- 1
        }else{
          dist2_valid <- t(apply(dist2_valid,1,function(x){x<-x-min(x)}))
          U_valid <- exp(-dist2_valid / vars[2])
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
    fn = cv_xie_beni_optim,
    method = "Nelder-Mead",
    folds = folds, 
    fold_ids = fold_ids,
    G = G,
    Q = Q,
    X = Xtr,
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
    
