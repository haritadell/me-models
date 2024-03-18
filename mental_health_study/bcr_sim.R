######### R script: BCRana ##########

# For performing penalised spline-based nonparametric 
# regression for classical measurement error, for
# analysis of actual data from Berry, Carroll & Ruppert 
# (Journal of the American Statistical Association, 2002).

# Last changed: 22 AUG 2017

# Set flag for code compilation (needed if 
# running script first time in current session) :

compileCode <- TRUE

# Load required packages:

library(HRW);  library(rstan)

# Set MCMC parameters:

nWarm <- 1000         # Length of burn-in.
nKept <- 1000           # Size of the kept sample.
nThin <- 1              # Thinning factor. 

# Set measurement error standard deviation:

sigmaW <- sqrt(0.35)

# Set the number of spline basis functions:

ncZ <- 30

# Load in data:

#data(BCR)
for (r in 1:100){
  cat('######## Now running r = ', r)
  data_r = read.csv(paste("~/Dropbox/sim_data_bcr/sim_data_bcr_", r, ".csv", sep=""))
  w <- data_r$w   ;   y <- data_r$y
  treatIndic <- as.numeric(as.character(data_r$status) == "treatment")
  n <- length(y)
  set.seed(100+r)
  
  # Specify model in Stan:
  
  BCRanaModel <- 
    'data
   {
      int<lower = 1> n;           int<lower = 1> ncZ;
      vector[n] y;              vector[n] w;        
      vector[n] treatIndic;              
      real<lower = 0> sigmaBeta;  real<lower = 0> sigmaMu;
      real<lower = 0> sigmaW;     
      real<lower = 0> Ax;         real<lower = 0> Aeps;
      real<lower = 0> Au;           
   }
   parameters 
   {
      vector[4] beta;       
      vector[ncZ] uControl;           vector[ncZ] uTreatmt;            
      vector[n] x;
      real muX;                       real<lower = 0> sigmaX;
      real<lower = 0> sigmaUcontrol;    real<lower = 0> sigmaUtreatmt;
      real<lower = 0> sigmaEps;   
   }
   transformed parameters 
   {
      matrix[n,4] X;            vector[ncZ] knots;
      matrix[n,ncZ] Z;
      for (k in 1:ncZ)
         knots[k] = ((ncZ+1-k)*min(x)+k*max(x))/(ncZ+1);
      for (i in 1:n)
      {
         X[i,1] = 1     ;   X[i,2] = treatIndic[i];
         X[i,3] = x[i]  ;   X[i,4] = treatIndic[i]*x[i];
         for (k in 1:ncZ)   
            Z[i,k] = (x[i]-knots[k])*step(x[i]-knots[k]);
      }
   }
   model 
   {
      for (i in 1:n)
         y[i] ~ normal((dot_product(beta,X[i])
                       + dot_product(uControl,((1-treatIndic[i])*Z[i]))
                       + dot_product(uTreatmt,(treatIndic[i]*Z[i]))),sigmaEps);
      x  ~ normal(muX,sigmaX); 
      w ~ normal(x,sigmaW);
      uControl ~ normal(0,sigmaUcontrol) ; uTreatmt ~ normal(0,sigmaUtreatmt); 
      beta ~ normal(0,sigmaBeta);
      muX ~ normal(0,sigmaMu); sigmaX ~ cauchy(0,Ax);       
      sigmaEps ~ cauchy(0,Aeps); 
      sigmaUcontrol ~ cauchy(0,Au); sigmaUtreatmt ~ cauchy(0,Au); 
   }'
  
  # Set up input data:
  
  allData <- list(n = n,ncZ = ncZ,w = w,y = y,treatIndic = treatIndic,
                  sigmaW = sigmaW,sigmaMu = 1e5,
                  sigmaBeta = 1e5,Ax = 1e5,Aeps = 1e5,Au = 1e5)
  
  # Compile code for model if required:
  
  if (compileCode)
    stanCompilObj <- stan(model_code = BCRanaModel,data = allData,
                          iter = 1,chains = 1)
  
  # Perform MCMC:
  
  stanObj <-  stan(model_code = BCRanaModel,data = allData,warmup = nWarm,
                   iter = (nWarm + nKept),chains = 1,thin = nThin,refresh = 25,
                   fit = stanCompilObj)
  
  # Extract relevant MCMC samples:
  
  betaMCMC <- NULL
  for (j in 1:4)
  {
    charVar <- paste("beta[",as.character(j),"]",sep = "") 
    betaMCMC <- rbind(betaMCMC,extract(stanObj,charVar,permuted = FALSE))
  }
  uControlMCMC <- NULL  ;  uTreatmtMCMC <- NULL
  for (k in 1:ncZ)
  {
    charVar <- paste("uControl[",as.character(k),"]",sep = "") 
    uControlMCMC <- rbind(uControlMCMC,extract(stanObj,charVar,permuted = FALSE))
    
    charVar <- paste("uTreatmt[",as.character(k),"]",sep = "") 
    uTreatmtMCMC <- rbind(uTreatmtMCMC,extract(stanObj,charVar,permuted = FALSE))
  }
  muXMCMC <- as.vector(extract(stanObj,"muX",permuted = FALSE))
  sigmaXMCMC <- as.vector(extract(stanObj,"sigmaX",permuted = FALSE))
  sigmaEpsMCMC <- as.vector(extract(stanObj,"sigmaEps",permuted = FALSE))
  sigmaUcontrolMCMC <- as.vector(extract(stanObj,"sigmaUcontrol",permuted = FALSE))
  sigmaUtreatmtMCMC <- as.vector(extract(stanObj,"sigmaUtreatmt",permuted = FALSE))
  xMCMC <- NULL
  for (i in 1:n)
  {
    charVar <- paste("x[",as.character(i),"]",sep = "") 
    xMCMC <- rbind(xMCMC,extract(stanObj,charVar,permuted = FALSE))
  }
  knotsMCMC <- NULL
  for (i in 1:ncZ)
  {
    charVar <- paste("knots[",as.character(i),"]",sep = "") 
    knotsMCMC <- rbind(knotsMCMC,extract(stanObj,charVar,permuted = FALSE))
  }
  
  thetas = rbind(betaMCMC, uControlMCMC, uTreatmtMCMC)
  
  write.table(thetas, paste("~/Dropbox/bcr_sample_", r, ".txt", sep=""))
  
  ############ End of BCRana ############
  
}




