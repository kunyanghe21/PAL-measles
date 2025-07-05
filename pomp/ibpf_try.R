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
  iota      = 0,
  betabar   = 6.32,
  c         = 0.219,
  a         = 0.1467,
  rho       = 0.142,
  gamma     = 0.0473,
  delta     = 0.02/(26*4),
  sigma_xi  = 0.318,
  gaussianrho     = 0.7,
  psi      = 0.306,
  g         = 716,
  S_0       = 0.02545,
  E_0       = 0.00422,
  I_0       = 0.000061
)

U  <- 40

shared_unit_values <- list(
  alpha        = 1,
  iota         = 0,
  c            = 0.032121729,
  a            = 0.37107605,
  rho          = 0.11784376,
  gamma        = 0.10496352,
  delta        = 0.0001923076923076923,
  sigma_xi     = 0.1776823,
  psi          = 0.20831605,
  g            = 621.00315
)


betabar_vals <- c(
  0.88894647, 0.59131747, 1.3571352,  0.99088466, 1.1142877,
  1.073081,   1.5383922,  0.90597922, 0.82007521, 0.85654908,
  1.0977113,  0.86888778, 1.1310883,  0.9737336,  0.94829279,
  0.62169701, 0.94519132, 0.59131724, 0.59131730, 0.76468575,
  1.188135,   0.96471751, 1.3417162,  1.2114047,  1.2141544,
  1.4035884,  0.98535007, 0.78903711, 1.3568828,  0.92531413,
  0.85022265, 1.1698065,  1.747846,   1.0385478,  0.94807833,
  0.90443295, 1.564422,   1.3209716,  1.4434845,  0.89985961
)

gaussianrho_vals <- c(
  0.512295, 0.5846346, 0.73733903, 0.64644552, 0.68464784,
  0.62805587, 0.59746622, 0.62227199, 0.57668756, 0.68815189,
  0.75385802, 0.63758144, 0.65418166, 0.59742807, 0.58928316,
  0.7143576, 0.64989739, 0.52764322, 0.51610781, 0.61995668,
  0.63451302, 0.64841047, 0.66641999, 0.58411126, 0.74216196,
  0.65110744, 0.57886699, 0.6156806, 0.51067983, 0.47119888,
  0.70101722, 0.59482185, 0.76027942, 0.66776662, 0.52625039,
  0.60342687, 0.68950329, 0.5869341,  0.70792855, 0.57809921
)

S0_vals <- c(
  0.075939171, 0.045746908, 0.072099209, 0.093366429, 0.12166313,
  0.082433015, 0.076253399, 0.073552363, 0.091899686, 0.11601865,
  0.11060591,  0.086007304, 0.090033337, 0.068562545, 0.08527638,
  0.099353261, 0.071033828, 0.057539228, 0.0457469,   0.056584992,
  0.066623405, 0.071289554, 0.10958349,  0.057944749, 0.098137014,
  0.083878987, 0.073387422, 0.065006606, 0.076856337, 0.089820623,
  0.078178115, 0.11018048,  0.1181622,   0.098707311, 0.09720967,
  0.076138817, 0.066983677, 0.081948154, 0.083776928, 0.063756049
)

E0_vals <- c(
  6.5060573e-05, 4.5087443e-05, 5.2817129e-05, 6.8182577e-05, 6.4530977e-05,
  4.4786735e-05, 7.5147931e-05, 9.4057032e-05, 4.7683639e-05, 4.2368429e-05,
  6.2710962e-05, 7.0720002e-05, 6.5743807e-05, 9.3865732e-05, 9.4957613e-05,
  5.9195183e-05, 4.9652379e-05, 4.4339824e-05, 5.5780311e-05, 3.4462537e-05,
  6.6237786e-05, 4.6263423e-05, 4.9870680e-05, 4.3129025e-05, 7.4057010e-05,
  6.1690727e-05, 7.0626847e-05, 7.1526090e-05, 5.3734158e-05, 6.6238754e-05,
  5.7956771e-05, 6.3070715e-05, 8.4001782e-05, 8.3680527e-05, 3.8972536e-05,
  5.9952090e-05, 5.8326987e-05, 5.7037920e-05, 8.9273613e-05, 6.1138249e-05
)

I0_vals <- c(
  1.25675637e-04, 1.28807835e-04, 0.0025939241, 0.0011983996, 5.3031814e-05,
  0.00043914493, 0.00035218208, 0.0020560939, 0.00058301724, 0.00017986108,
  7.6727294e-05, 6.1292536e-05, 0.00021765193, 0.00083288341, 1.3892895e-08,
  0.0016802071, 0.0013289195, 0.00024760532, 0.00020639459, 0.0014469447,
  0.00066906348, 0.00023718234, 1.6463227e-08, 0.00067811750, 0.00091942732,
  0.00076855475, 0.0029490807, 0.0024151804, 0.0013736223, 0.0010850356,
  0.0030775415, 0.0039083785, 0.00043130366, 0.0025644207, 0.0023132868,
  0.00070547545, 0.00088096713, 0.00051698624, 0.00079949765, 0.00021292546
)


expand_units <- function(value, name) {
  vals <- if (length(value) == 1) rep(value, U) else value
  setNames(vals, paste0(name, seq_len(U)))
}

basic_paramsC <- c(
  unlist(lapply(names(shared_unit_values), function(nm)
    expand_units(shared_unit_values[[nm]], nm)
  )),
  
  expand_units(betabar_vals,     "betabar"),
  expand_units(gaussianrho_vals, "gaussianrho"),
  
  expand_units(S0_vals, "S_0"),
  expand_units(E0_vals, "E_0"),
  expand_units(I0_vals, "I_0")
)

length(basic_paramsC)          
head(basic_paramsC, 20)       


expandedParNames <- names(basic_params)

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

coef(m9) <- basic_paramsC

sim <- simulate(m9)

## ----bm-model,echo=T,eval=T---------------------------------------------------
library(spatPomp)
i <- 2

measles_dir <- paste0("measles_more",i,"/")
if(!dir.exists(measles_dir)) dir.create(measles_dir)

## -------- new rw.sd specification ----------------------------------
par_rw <- setdiff(names(basic_params), c("alpha", "iota","detla","gaussianrho","psi")) 

U       <- 40          
ivp_sd  <- 0.001        
rp_sd   <- 0.001       

IVP_names   <- unlist(lapply(c("S_0", "E_0", "I_0"),
                             function(p) paste0(p, 1:U)))

OTHER_names <- unlist(lapply(setdiff(par_rw, c("S_0", "E_0", "I_0")),
                             function(p) paste0(p, 1:U)))


string_rwsd <- paste0(
  "rw_sd(",
  paste0(IVP_names,   "=ivp(", ivp_sd, ")", collapse = ", "),
  if (length(OTHER_names) > 0)
      paste0(", ", paste0(OTHER_names, "=", rp_sd, collapse = ", ")),
  ")"
)

measles_rw.sd <- eval(parse(text = string_rwsd))

## ----ibpf-mle-eval,eval=T,echo=F----------------------------------------------
stew(file=paste0(measles_dir,"ibpf_mle.rda"),seed=999,{
  tic <- Sys.time()
  params_start <- coef(m9)
  ibpf_mle_searches <- foreach(reps=1:switch(i,3,10))%dopar%{
    ibpf(m9,params=params_start,
      Nbpf=switch(i,2,50),Np=switch(i,10,10000),
      rw.sd=measles_rw.sd ,
      unitParNames= c("betabar","S_0","E_0","I_0") ,
      sharedParNames = c("a","c","rho","gamma","sigma_xi","g"),
      block_size=1,
      spat_regression=0.1,
      cooling.fraction.50=0.5
    )
  }
  toc <- Sys.time()
  })

  prof1time <- toc-tic

local_search <- bake(
  file = paste0(measles_dir, "local_search.rda"),  # => measles_more2/local_search.rda
  {
    foreach(mf = ibpf_mle_searches, .combine = rbind) %dopar% {
      evals <- replicate(10, logLik(bpfilter(mf, Np = 20000, block_size = 1)))
      ll    <- logmeanexp(evals, se = TRUE)
      mf %>% coef() %>%
        bind_rows() %>%                     
        bind_cols(loglik = ll[1],          
                  loglik.se = ll[2])        
    }
  }
)

m1 <- ibpf_mle_searches[[9]]

sim <- simulate(m1)

plot(sim,log = T)  

tmp <- bpfilter(m1,Np = 20000,block_size = 1)

## ----traces,fig.height=6.5, fig.width=6.5, out.width="6.5in", fig.cap = "Trace plots for the correlated Gaussian random walk example."----
ibpf_traces <- pomp::melt(lapply(ibpf_mle_searches,pomp:::traces_internal))
ibpf_traces$iteration <- as.numeric(ibpf_traces$iteration)
# Define ma thematical labels for facet_wrap
param_labels <- c(
  loglik   = "logLik"
)

ibpf_log <- ibpf_traces %>%
  filter(variable == "loglik")

# Generate the plot
ggplot(ibpf_log, aes(x=iteration, y=value, group=.L1, color=factor(.L1))) +
  geom_line(size = 0.1) +
  guides(color="none")
