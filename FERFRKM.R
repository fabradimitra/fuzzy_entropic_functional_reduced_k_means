FERFRKM <- function(C,K,Pk,Lk,U,A,B,lambda,gamma,max_iter = Inf,tol = 1e-6){
  # This function implements the FEFRKM algorithm for fuzzy functional data clustering.
  # Inputs:
  #  C: (I x J) coefficients for the natural cubic spline basis functions for each individual i
  #  K: (J x J) inner products of second derivative of the natural cubic spline basis functions 
  #  U: (I x G) initial fuzzy membership matrix
  #  A: (G x Q) initial coefficients of the cluster centroid expansion in the reduced subspace
  #  B: (J x Q) initial coefficients of the basis of the reduced subspace wrt the natural cubic 
  #  spline basis functions
  #  lambda: roughness penalty parameter for the cluster centroids
  #  gamma: degree of fuzziness
  #  max_iter: maximum number of iterations
  #  tol: tolerance for convergence
  #######################################################################################
  I <- nrow(C) 
  J <- ncol(C) 
  G <- ncol(U) 
  Q <- ncol(B)
  stopifnot(is.matrix(C), is.matrix(K), is.matrix(U), is.matrix(A), is.matrix(B))
  if (!all(dim(K) == c(J, J))) stop("K must be J x J with J = ncol(C).")
  if (!all(dim(U) == c(I, G))) stop("U must be I x G with I = nrow(C).")
  if (!all(dim(A) == c(G, Q))) stop("A must be G x Q with G = ncol(U).")
  if (!all(dim(B) == c(J, Q))) stop("B must be J x Q with J = ncol(C) and Q = ncol(A).")
  if (!isTRUE(all.equal(K, t(K), tol = 1e-10))) stop("K must be symmetric.")
  if (lambda < 0 || gamma < 0 || tol <= 0 || max_iter < 1) stop("Invalid hyperparameters.")
  if (any(!is.finite(C)) || any(!is.finite(K)) || any(!is.finite(U)) || any(!is.finite(A)) || any(!is.finite(B))) {
  stop("Inputs contain NA/NaN/Inf.")
  }
  cnorm2 <- rowSums(C*C) # vector of lenght J having squared norm of c_i as element
  IJ <- diag(J)
  IG <- diag(G)
  D <- diag(sqrt(colSums(U)))
  D2 <- D^2
  Cbar <- diag(1/diag(D2))%*%t(U)%*%C
  #
  loss_function_curr <- Inf
  dif <- Inf
  iter <- 0
  while(dif > tol & iter < max_iter){
    iter <- iter + 1
    A <- A/max(A)
    # Update B
    B <- solve(kronecker(t(A)%*%D2%*%A,IJ)+lambda*kronecker(t(A)%*%A,K))%*%c(t(Cbar)%*%D2%*%A)
    B <- matrix(B, nrow = J, ncol = Q)
    #######################################################################################
    # Update A
    A <- solve(kronecker(t(B)%*%B,D2)+lambda*kronecker(t(B)%*%K%*%B,IG))%*%c(D2%*%Cbar%*%B)
    A <- matrix(A, nrow = G, ncol = Q)
    #######################################################################################
    # Update U
    Distxmg2 <- t(matrix(rowSums((C%x%c(rep(1,G))-c(rep(1,I))%x%(A%*%t(B)))^2),G,I))
    if(gamma==0){
      Un <- matrix(0,nrow = I, ncol = G)
      Un[cbind(seq_len(I), max.col(-Distxmg2, ties.method = "first"))] <- 1
      su <- colSums(Un)
      if(!any(su==0)){U <- Un}
    }else{
      Distxmg2 <- t(apply(Distxmg2,1,function(x){x<-x-min(x)}))
      U <- exp(-Distxmg2/gamma)
      U <- U/matrix(rowSums(U),I,G)
    }
    D <- diag(sqrt(colSums(U)))
    D2 <- D^2
    Cbar <- diag(1/diag(D2))%*%t(U)%*%C
    #######################################################################################
    # Compute loss function and check convergence
    loss_function_new <- loss_function(U, C, Cbar, D, A, B, K, lambda, gamma)
    dif <-  loss_function_curr - loss_function_new$lossp
    loss_function_curr <- loss_function_new$lossp
    if(iter%%100==0){
    cat("Iteration: ", iter, " Loss pen: ", loss_function_curr, " Loss: ", loss_function_new$loss,
    " Difference: ", dif, " Norm B: ", norm(B, type = "F")," Norm A: ", norm(A, type = "F"), "\n")
    }}
  cat("Iteration: ", iter, " Loss pen: ", loss_function_curr, " Loss: ", loss_function_new$loss,
      " Difference: ", dif, " Norm B: ", norm(B, type = "F")," Norm A: ", norm(A, type = "F"), "\n")
  return(list(U = U, A = A, B = B, Cbar = Cbar,
    loss_function = loss_function_curr, 
    loss_function_unpen = loss_function_new$loss,
    wdev = loss_function_new$wdev,
    iterations = iter))
} 