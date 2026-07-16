library(mclust)

k2m <- function(m1,m2,x){
  dif <- 1
  while (dif>1.0e-9){
    tt <- (m1+m2)/2
    m1n <- mean(x[x<tt])
    m2n <- mean(x[x>tt])
    dif <- sum((m1n-m1)^2+(m2n-m2)^2)
    m1 <- m1n
    m2 <- m2n
  }
  return(c(m1,m2))
}

fek2m <- function(m1,m2,x){
  dif <- 1
  while (dif>1.0e-9){
    w1 <- exp(-0.5*(x-m1)^2)
    w2 <- exp(-0.5*(x-m2)^2)
    p1 <- w1/(w1+w2)
    p2 <- w2/(w1+w2)
    m1n <- sum(p1*x)/sum(p1)
    m2n <- sum(p2*x)/sum(p2)
    dif <- sum((m1n-m1)^2+(m2n-m2)^2)
    m1 <- m1n
    m2 <- m2n
  }
  return(sort(c(m1,m2)))
}

set.seed(123)
Res <- matrix(0,100,6)
for(i in 1:100){
  n <- 1000
  mu <- c(0.6,1.8)
  x <- rep(mu,n/2)+rnorm(n,sd=6)
  # try the different starting points
  #mu0 <- mu
  #mu0 <- c(0.4,1.6)
  #mu0 <- sort(rnorm(2))
  #mu0 <- c(mean(x[x<(mean(x)+0.5*sd(x))]),mean(x[x>(mean(x)-0.5*sd(x))]))
  mu0 <- c(quantile(x,.25),quantile(x,.75))
  mu1 <- k2m(mu0[1],mu0[2],x)
  op <- Mclust(x,G=2,modelName="E",verbose=FALSE)
  mu3 <- op$parameters$mean
  Res[i,] <- c(mu1,fek2m(mu0[1],mu0[2],x),mu3)-c(mu,mu,mu)
}
plot(Res[,1],ylim=c(-1.2,1.2)); points(Res[,3],col="red")
colSums(Res^2)
