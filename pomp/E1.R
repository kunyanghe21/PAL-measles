## ----packages,incluxde=F,echo=F,cache=F----------------------------------------
library("spatPomp")
library("ggplot2")
library("tidyverse")
library("knitr")
library("doRNG")
library("doParallel")

pomp_dir="pomp/"


library(pomp)

measles_cases <- read.csv(paste0(pomp_dir,"case1.csv"))
measles_covar <- read.csv(paste0(pomp_dir,"covar2.csv"))

measles_cases<- measles_cases[measles_cases$city == "LONDON", ]
measles_covar <- measles_covar[measles_covar$city == "LONDON", ]


measles_cases <-  measles_cases[,-1]
measles_covar <-  measles_covar[,-1]



colnames(measles_cases) <- c("time","cases1")
colnames(measles_covar) <- c("time",
                             "lag_birthrate1","pop1")


basic_params <- c(
  alpha       = 1,
  iota        = 0,
  betabar     = 6.32,
  c           = 0.219,
  a           = 0.1476,
  rho         = 0.142,
  gamma       = 0.0473,
  delta       = 0.02/(26*4),  # timescale transform
  sigma_xi    = 0.318,
  gaussianrho = 0.55,
  psi         = 0.306,
  g           = 0,
  S_0         = 0.02545,
  E_0         = 0.00422,
  I_0         = 0.000061
)


rproc <- Csnippet("
  double t_mod = fmod(t, 364.0);
  double br1;
  double beta1, seas1;
  double foi1;         
  double xi1;           
  double betafinal1;

  int trans_S1[2], trans_E1[2], trans_I1[2];
  double prob_S1[2], prob_E1[2], prob_I1[2];

  if ((t_mod >= 6 && t_mod < 99) ||
      (t_mod >= 115 && t_mod < 198) ||
      (t_mod >= 252 && t_mod < 299) ||
      (t_mod >= 308 && t_mod < 355)) {
    seas1 = 1.0 + a * 2 * (1 - 0.759);
  } else {
    seas1 = 1.0 - 2 * a * 0.759;
  }

  beta1 = betabar * seas1;

  if (fabs(t_mod - 248.5) < 0.5) {
    br1 = c * lag_birthrate1;
  } else {
    br1 = (1.0 - c) * lag_birthrate1 / 103.0;
  }

  double I_ratio1 = I1 / pop1;

  foi1 = pow((I1 + iota) / pop1, alpha);
 
  xi1 = rgamma(sigma_xi, 1 / sigma_xi);;
  betafinal1 = beta1 * I_ratio1 * xi1;

  int SD1 = rbinom(S1, delta);
  int ED1 = rbinom(E1, delta);
  int ID1 = rbinom(I1, delta);
  int RD1 = rbinom(R1, delta);

  S1 -= SD1;  E1 -= ED1;  I1 -= ID1;  R1 -= RD1;
  
  prob_S1[0] = exp(-dt * betafinal1);
  prob_S1[1] = 1 - exp(-dt * betafinal1);

  prob_E1[0] = exp(-dt * rho);
  prob_E1[1] = 1 - exp(-dt * rho);

  prob_I1[0] = exp(-dt * gamma);
  prob_I1[1] = 1 - exp(-dt * gamma);

  rmultinom(S1, prob_S1, 2, trans_S1);
  rmultinom(E1, prob_E1, 2, trans_E1);
  rmultinom(I1, prob_I1, 2, trans_I1);

  S1 = trans_S1[0] + rpois(br1);
  E1 = trans_E1[0] + trans_S1[1];
  I1 = trans_I1[0] + trans_E1[1];
  R1 += trans_I1[1];
  C1 += trans_I1[1];
");




## ----dmeasure-------------------------------------------------
dmeas <- Csnippet("
  double m = gaussianrho*C1;
  double v = m*(1.0-gaussianrho+psi*psi*m);
  double tol = 0.0;
  if (cases1 > 0.0) {
    lik = pnorm(cases1+0.5,m,sqrt(v)+tol,1,0)
           - pnorm(cases1-0.5,m,sqrt(v)+tol,1,0) + tol;
  } else {
    lik = pnorm(cases1+0.5,m,sqrt(v)+tol,1,0) + tol;
  }
  if (give_log) lik = log(lik);
")

## ----rmeasure-------------------------------------------------
rmeas <- Csnippet("
  double m = gaussianrho*C1;
  double v = m*(1.0-gaussianrho+psi*psi*m);
  double tol = 0.0;
  cases1 = rnorm(m,sqrt(v)+tol);
  if (cases1 > 0.0) {
    cases1 = nearbyint(cases1);
  } else {
    cases1 = 0.0;
  }
")

rinit <- Csnippet("
  double probs1[4];
  probs1[0] = S_0;
  probs1[1] = E_0;
  probs1[2] = I_0;
  probs1[3] = 1.0 - probs1[0] - probs1[1] - probs1[2];

  int counts1[4];
  rmultinom(pop1, probs1, 4, counts1);

  S1 = counts1[0];
  E1 = counts1[1];
  I1 = counts1[2];
  R1 = counts1[3];
  C1 = 0;
");

basic_log_names   <- c("rho", "gamma", "sigma_xi", "betabar", "g", "iota", "delta")
basic_logit_names <- c("a", "alpha", "c", "gaussianrho", "S_0", "E_0", "I_0", "psi")
log_names   <- basic_log_names
logit_names <- basic_logit_names
measles_partrans <- parameter_trans(
  log   = log_names,
  logit = logit_names
)

one_city_pomp <- pomp(
  data       = measles_cases,
  times      = "time",
  t0         = 0,
  rprocess   = euler(rproc, delta.t = 3.5), 
  rinit      = rinit,
  dmeasure   = dmeas,
  rmeasure   = rmeas,
  statenames = c("S1","E1","I1","R1","C1"),
  paramnames = c("alpha","iota","betabar","c","a","rho","gamma",
                 "delta","sigma_xi","g","gaussianrho","psi",
                 "S_0","E_0","I_0"),
  covar      = covariate_table(measles_covar,times = "time"),
  covarnames = c("lag_birthrate1","pop1"),
  accumvars  = c("C1")
)

coef(one_city_pomp) <- basic_params

sim <- simulate(one_city_pomp, params =  basic_params,  nsim   = 1,
                seed   = 154234)

plot(sim)

pf_result <- pfilter(sim, Np=5000)

pf_result@loglik


Pomp_dir <- paste0(pomp_dir,"Pomp_E",1,"/")
if(!dir.exists(Pomp_dir)) dir.create(Pomp_dir)

stew(file=paste0(Pomp_dir,"E1_non_optimize.rda"),seed=456,{
  
  cat(capture.output(sessionInfo()),
      file=paste0(Pomp_dir,"sessionInfo.txt"),sep="\n")
  
  pf_logLik <- replicate(20,
                         logLik(pfilter(sim,Np = 100000))
  )
  
  
})

pf_logLik

result <- logmeanexp(pf_logLik,se = T)

tmp_mean <- mean(pf_logLik)

tmp_var <- var(pf_logLik)

E1_sim <- sim@data

E1_sim <- t(E1_sim)

negloglik <- function(x) optim(par=c(0.5,0.5,1),function(theta)-sum(dnbinom(x,mu=theta[1]+theta[2]*c(0,head(x,-1)),size=theta[3],log=T)))$value

tmp_negbinom <- -sum(apply(E1_sim,2,negloglik))



library(tidyverse)
library(pomp)
library(foreach)
library(doFuture)
library(doRNG)


set.seed(1350254336)

one_city_pomp |>
  pfilter(Np=1000) -> pf  
logLik(pf)  

plan(multisession)  
registerDoRNG(seed=999) 

tic <- Sys.time()
foreach(i=1:10, .combine=c, .options.future=list(seed=TRUE)) %dofuture% {
  one_city_pomp |>
    pfilter(Np=5000)   
} -> pf_list

pf_list |> logLik() |> logmeanexp(se=TRUE) -> L_pf
L_pf

toc <- Sys.time()

bake(file="local_search_E1*.rds",{
  foreach(i=1:10,.combine=c,
          .options.future=list(seed=482947940)
  ) %dofuture% {
    one_city_pomp |>
      mif2(
        Np=20000, Nmif=50,
        cooling.fraction.50=0.5,
        rw.sd=rw_sd(c        = 0.02,
                    betabar  = 0.02,
                    a        = 0.02,
                    rho      = 0.02,
                    gamma    = 0.02,
                    sigma_xi = 0.02,
                    psi     = 0.02,
                    gaussianrho = 0.02,
                    S_0      = ivp(0.02),
                    E_0      = ivp(0.02),
                    I_0      = ivp(0.02)),
        partrans=parameter_trans(log=c("rho", "gamma", "sigma_xi", "betabar", "g", "iota", "delta"),logit = c("a", "alpha", "c", "gaussianrho", "S_0", "E_0", "I_0", "psi")),
        paramnames=c("alpha","iota","betabar","c","a","rho","gamma",
                     "delta","sigma_xi","g","gaussianrho","psi",
                     "S_0","E_0","I_0")
      )
  } -> mifs_local
  attr(mifs_local,"ncpu") <- nbrOfWorkers()
  mifs_local
}) -> mifs_local
t_loc <- attr(mifs_local,"system.time")
ncpu_loc <- attr(mifs_local,"ncpu")


mifs_local |>
  traces() |>
  melt() |>
  ggplot(aes(x = iteration, y = value, group = .L1, color = factor(.L1))) +
  geom_line() +
  guides(color = "none") +
  facet_wrap(~ name, scales = "free_y")


local_search <- foreach(
  mf = mifs_local,.combine = rbind) %dopar% { 
    evals <- replicate(20, logLik(pfilter(mf,Np=5000)))
    ll <- logmeanexp(evals, se=TRUE)
    mf %>% coef() %>% bind_rows() %>%
      bind_cols(loglik=ll[1],loglik.se=ll[2])
  }


bind_rows(local_search) %>%
  filter(is.finite(loglik)) %>%
  filter(loglik.se < 2) %>%
  arrange(-loglik) -> best_searches    

head(best_searches)

best_searches <- best_searches %>%
  select(-loglik, -loglik.se)

best_param <- best_searches[1,]

coef(one_city_pomp) <- best_param

sim <- simulate(
  one_city_pomp,
  params = best_param,
  nsim   = 1,
  seed   = 154234
)

plot(sim)

pf_result <- pfilter(sim, Np=5000)

pf_result@loglik

sim.data <- as.data.frame(sim)

londonsim <- sim.data$cases1

df <- as.data.frame(t(londonsim)) 

colnames(df) <- 0:(length(londonsim) - 1)

write.csv(df, "londonsim.csv", row.names = FALSE)


Pomp_dir <- paste0("Pomp_E",1,"/")
if(!dir.exists(Pomp_dir)) dir.create(Pomp_dir)

stew(file=paste0(Pomp_dir,"E1.rda"),seed=456,{
  
  cat(capture.output(sessionInfo()),
      file=paste0(Pomp_dir,"sessionInfo.txt"),sep="\n")
  
  pf_logLik <- replicate(20,
                           logLik(pfilter(sim,Np = 5000))
  )
  
  
})

pf_logLik
