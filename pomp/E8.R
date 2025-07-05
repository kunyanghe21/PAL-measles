## ----packages,incluxde=F,echo=F,cache=F----------------------------------------
## The spatPomp code is primarily adapted from Ionides et al. (2024) and the Whitehouse et al.\ (2023) code implementation.

library("spatPomp")
library("ggplot2")
library("tidyverse")
library("knitr")
library("doRNG")
library("doParallel")
cores <-  as.numeric(Sys.getenv('SLURM_NTASKS_PER_NODE', unset=NA))
if(is.na(cores)) cores <- detectCores()
registerDoParallel(cores)
ggplot2::theme_set(ggplot2::theme_bw())

stopifnot(packageVersion("pomp")>="5.0")

pomp_dir="pomp/"

set.seed(42)

basic_params <- c(
  alpha     = 1,
  iota      = 0.1,
  betabar   = 6.32,
  c         = 0.219,
  a         = 0.1467,
  rho       = 0.142,
  gamma     = 0.0473,
  delta     = 0.02/(26*4),
  sigma_xi  = 0.318,
  gaussianrho     = 0.7,
  psi      = 0.306,
  g         = 0,
  S_0       = 0.02545,
  E_0       = 0.00422,
  I_0       = 0.000061
)
expandedParNames <- NULL

dt <- 3.5
U  <- 40

measles_cases  <- read.csv(paste0(pomp_dir,"case1.csv"))

measles_covar  <- read.csv(paste0(pomp_dir,"covar2.csv"))

measles_covarnames <- paste0(rep(c("pop", "lag_birthrate"), each = U), 1:U)
measles_unit_covarnames <- c("pop", "lag_birthrate")

data_measles_distance <- read.csv(paste0(pomp_dir,'data_measles_distance.csv'))




data_measles_distance <- data_measles_distance

v_by_g <- as.matrix(data_measles_distance)

to_C_array <- function(v) paste0("{", paste0(v, collapse = ","), "}")
v_by_g_C_rows  <- apply(v_by_g, 1, to_C_array)
v_by_g_C_array <- to_C_array(v_by_g_C_rows)
v_by_g_C <- Csnippet(paste0("const double v_by_g[", U, "][", U, "] = ", v_by_g_C_array, "; "))

parNames      <- names(basic_params)
fixedParNames <- setdiff(parNames, expandedParNames)

set_expanded <- Csnippet(
  paste0("const int ", expandedParNames, "_unit = 1;\n", collapse = " ")
)
set_fixed <- Csnippet(
  paste0("const int ", fixedParNames, "_unit = 0;\n", collapse = " ")
)
measles_globals <- Csnippet(
  paste(v_by_g_C, set_expanded, set_fixed, sep = "\n")
)

measles_paramnames <- c(
  if (length(fixedParNames) > 0) {
    paste0(fixedParNames, "1")
  },
  if (length(expandedParNames) > 0) {
    paste0(rep(expandedParNames, each = U), 1:U)
  }
)

unit_statenames <- c("S", "E", "I", "R", "C")


measles_rprocess <- spatPomp_Csnippet(
  unit_statenames  = c("S", "E", "I", "R", "C"),
  unit_covarnames  = c("pop", "lag_birthrate"),
  unit_paramnames  = c("alpha", "iota", "betabar", "c", "a",
                       "rho", "gamma", "delta", "sigma_xi", "g"),
  code ="
    // Variables
    double br, beta, seas, foi, births, xi, betafinal;
    int trans_S[2], trans_E[2], trans_I[2];
    double prob_S[2], prob_E[2], prob_I[2];
    int SD[U], ED[U], ID[U], RD[U];
    double powVec[U];
    int u, v;

    // Calculate the day of the year without any offset
    // Pre-computing this saves substantial time
    // powVec[u] = pow(I[u]/pop[u], alpha);
    for (u = 0; u < U; u++) {
        powVec[u] = I[u] / pop[u];
        // IS THIS INTENDED TO BE FIXED TO ALPHA=1?
    }

    for (u = 0; u < U; u++) {
        double t_mod = fmod(t, 364.0);

        // Transmission rate
        if ((t_mod >= 6 && t_mod < 99) || (t_mod >= 115 && t_mod < 198) ||
            (t_mod >= 252 && t_mod < 299) || (t_mod >= 308 && t_mod < 355))
            seas = 1.0 + a[u * a_unit] * 2 * (1 - 0.759);
        else
            seas = 1.0 - 2 * a[u * a_unit] * 0.759;

        beta = betabar[u * betabar_unit] * seas;

        // Birth rate calculation
        if (fabs(t_mod - 248.5) < 0.5) {
            br = c[u * c_unit] * lag_birthrate[u];
        } else {
            br = (1.0 - c[u * c_unit]) * lag_birthrate[u] / 103;
        }

        // Expected force of infection
        if (alpha[u * alpha_unit] == 1.0 && iota[u * iota_unit] == 0.0) {
            foi = I[u] / pop[u];
        } else {
            foi = pow((I[u] + iota[u * iota_unit]) / pop[u], alpha[u * alpha_unit]);
        }

        for (v = 0; v < U; v++) {
            if (v != u) {
                foi += g[u * g_unit] * v_by_g[u][v] * (powVec[v] - powVec[u]) / pop[u];
            }
        }

        xi = rgamma(sigma_xi[u * sigma_xi_unit], 1 / sigma_xi[u * sigma_xi_unit]);
        betafinal = beta * foi * xi;  // Stochastic force of infection

        // Poisson births
        births = rpois(br);

        SD[u] = rbinom(S[u], delta[u * delta_unit]);
        ED[u] = rbinom(E[u], delta[u * delta_unit]);
        ID[u] = rbinom(I[u], delta[u * delta_unit]);
        RD[u] = rbinom(R[u], delta[u * delta_unit]);

        S[u] = S[u] - SD[u];
        E[u] = E[u] - ED[u];
        I[u] = I[u] - ID[u];
        R[u] = R[u] - RD[u];

        // Probabilities for state transitions
        prob_S[0] = exp(-dt * betafinal);
        prob_S[1] = 1 - exp(-dt * betafinal);

        prob_E[0] = exp(-dt * rho[u * rho_unit]);
        prob_E[1] = 1 - exp(-dt * rho[u * rho_unit]);

        prob_I[0] = exp(-dt * gamma[u * gamma_unit]);
        prob_I[1] = 1 - exp(-dt * gamma[u * gamma_unit]);

        // Multinomial transitions
        rmultinom(S[u], &prob_S[0], 2, &trans_S[0]); // B, (S-F)-B
        rmultinom(E[u], &prob_E[0], 2, &trans_E[0]); // C, (E-F)-C
        rmultinom(I[u], &prob_I[0], 2, &trans_I[0]); // E, (I-F)-D

        // Update compartments
        S[u] = trans_S[0] + births;
        E[u] = trans_E[0] + trans_S[1];
        I[u] = trans_I[0] + trans_E[1];
        R[u] = R[u] + trans_I[1];
        C[u] += trans_I[1];  // True incidence
    }
"
)


measles_dmeasure <-  spatPomp_Csnippet(
  unit_statenames = 'C',
  unit_obsnames = 'cases',
  unit_paramnames = c('gaussianrho','psi'),
  code="
      double m,v;
      double tol = 1e-300;
      double mytol = 1e-5;
      int u;
      lik = 0;
      for (u = 0; u < U; u++) {
        m = gaussianrho[u*gaussianrho_unit]*(C[u]+mytol);
        v = m*(1.0-gaussianrho[u*gaussianrho_unit]+psi[u*psi_unit]*psi[u*psi_unit]*m);

        // Deal with NA measurements by omitting them
        if(!(ISNA(cases[u]))){
          // C < 0 can happen in bootstrap methods such as bootgirf
          if (C[u] < 0) {lik += log(tol);} else {
            if (cases[u] > tol) {
              lik += log(pnorm(cases[u]+0.5,m,sqrt(v)+tol,1,0)-
                pnorm(cases[u]-0.5,m,sqrt(v)+tol,1,0)+tol);
            } else {
                lik += log(pnorm(cases[u]+0.5,m,sqrt(v)+tol,1,0)+tol);
            }
          }
        }
      }
      if(!give_log) lik = (lik > log(tol)) ? exp(lik) : tol;
    "
)

measles_rmeasure <- spatPomp_Csnippet(
  method='rmeasure',
  unit_paramnames=c('gaussianrho','psi'),
  unit_statenames='C',
  unit_obsnames='cases',
  code="
      double m,v;
      double tol = 1.0e-300;
      int u;
      for (u = 0; u < U; u++) {
        m = gaussianrho[u*gaussianrho_unit]*(C[u]+tol);
        v = m*(1.0-gaussianrho[u*gaussianrho_unit]+psi[u*psi_unit]*psi[u*psi_unit]*m);
        cases[u] = rnorm(m,sqrt(v)+tol);
        if (cases[u] > 0.0) {
          cases[u] = nearbyint(cases[u]);
        } else {
          cases[u] = 0.0;
        }
      }
    "
)

measles_dunit_measure <- spatPomp_Csnippet(
  unit_paramnames=c('gaussianrho','psi'),
  code="
      double mytol = 1e-5;
      double m = gaussianrho[u*gaussianrho_unit]*(C+mytol);
      double v = m*(1.0-gaussianrho[u*gaussianrho_unit]+psi[u*psi_unit]*psi[u*psi_unit]*m);
      double tol = 1e-300;
      // C < 0 can happen in bootstrap methods such as bootgirf
      if(ISNA(cases)) {lik=1;} else { 
        if (C < 0) {lik = 0;} else {
          if (cases > tol) {
            lik = pnorm(cases+0.5,m,sqrt(v)+tol,1,0)-
              pnorm(cases-0.5,m,sqrt(v)+tol,1,0)+tol;
          } else {
            lik = pnorm(cases+0.5,m,sqrt(v)+tol,1,0)+tol;
          }
        }
      }
      if(give_log) lik = log(lik);
    "
)

measles_rinit <- spatPomp_Csnippet(
  unit_paramnames = c("S_0", "E_0", "I_0"),
  unit_statenames = c("S", "E", "I", "R", "C"),
  unit_covarnames = "pop",
  code = "
    int u;
    for (u = 0; u < U; u++) {
        double probs[4];
        probs[0] = S_0[u * S_0_unit];
        probs[1] = E_0[u * E_0_unit];
        probs[2] = I_0[u * I_0_unit];
        probs[3] = 1.0 - probs[0] - probs[1] - probs[2];
        int counts[4];
        rmultinom(pop[u], &probs[0], 4, &counts[0]);
        S[u] = counts[0];
        E[u] = counts[1];
        I[u] = counts[2];
        R[u] = counts[3];
        C[u] = 0;
    }
")




### === Parameter Transformation Settings ===

basic_log_names   <- c("rho", "gamma", "sigma_xi", "betabar", "g", "iota", "delta")
basic_log_names   <- setdiff(basic_log_names, fixedParNames)

basic_logit_names <- c("a", "alpha", "c", "gaussianrho", "S_0", "E_0", "I_0",'psi')
basic_logit_names <- setdiff(basic_logit_names, fixedParNames)
log_names   <- unlist(lapply(basic_log_names, function(x, U) paste0(x, 1:U), U))
logit_names <- unlist(lapply(basic_logit_names, function(x, U) paste0(x, 1:U), U))
measles_partrans <- parameter_trans(log = log_names, logit = logit_names)

m9 <- spatPomp(
  measles_cases,
  units           = "city",
  times           = "days",
  t0              = min(measles_cases$days) - 14,
  unit_statenames = unit_statenames,
  covar           = measles_covar,
  rprocess        = euler(measles_rprocess, delta.t = 3.5),
  unit_accumvars  = c("C"),
  paramnames      = measles_paramnames,
  globals         = measles_globals,
  rinit           = measles_rinit,
  dmeasure        = measles_dmeasure,
  rmeasure        = measles_rmeasure,
  dunit_measure   = measles_dunit_measure,
  partrans = measles_partrans
)

measles_params <- rep(0, length = length(measles_paramnames))

names(measles_params) <- measles_paramnames

for (p in fixedParNames)
  measles_params[paste0(p, 1)] <- basic_params[p]
for (p in expandedParNames)
  measles_params[paste0(p, 1:U)] <- basic_params[p]
coef(m9) <- measles_params

sim <- simulate(m9, params =  measles_params,  nsim   = 1,
                seed   = 154234)
##


spatPomp_dir <- paste0(pomp_dir,"E_",8,"/")
if(!dir.exists(spatPomp_dir)) dir.create(spatPomp_dir)

stew(file=paste0(spatPomp_dir,"E8.rda"),seed=124,{
  cat(capture.output(sessionInfo()),
      file=paste0(spatPomp_dir,"sessionInfo.txt"),sep="\n")
  
  bpf_logLik_40 <- foreach(i = 1:20, .combine = c) %dopar% {
    logLik(bpfilter(sim, Np = 100000, block_size = 1))
  }
})


E8_result <- logmeanexp(bpf_logLik_40,se = T,ess = T)

tmp_benchmark_spat <- arma_benchmark(sim)

tmp_benchmark_spat$total

E8_sim <- sim@data

E8_sim <- t(E8_sim)

negloglik <- function(x) optim(par=c(0.5,0.5,1),function(theta)-sum(dnbinom(x,mu=theta[1]+theta[2]*c(0,head(x,-1)),size=theta[3],log=T)))$value

tmp_negbinom_spat <- -sum(apply(E8_sim,2,negloglik))

realdata_benchmark_spat <- arma_benchmark(m9)

realdata_benchmark_spat$total

E8_real <- m9@data

E8_real <- t(E8_real)

realdata_negbinom_spat <- -sum(apply(E8_real,2,negloglik))

realdata_negbinom_spat

## Prepare the simulated data for python.
simdata <- as.data.frame(sim)

simdata <- simdata[order(simdata$city),]

yt <- simdata$cases

M40 <- matrix(yt, nrow = 40, byrow = TRUE)

M40 <- as.data.frame(M40)

colnames(M40) <- as.character(0:415)

write.csv(M40,file = "M40.csv",row.names = F)




