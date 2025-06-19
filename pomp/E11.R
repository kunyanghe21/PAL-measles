## ----packages,incluxde=F,echo=F,cache=F----------------------------------------
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

basic_paramsC <- c(
  alpha1 = 1,
  alpha2 = 1,
  alpha3 = 1,
  alpha4 = 1,
  alpha5 = 1,
  alpha6 = 1,
  alpha7 = 1,
  alpha8 = 1,
  alpha9 = 1,
  alpha10 = 1,
  alpha11 = 1,
  alpha12 = 1,
  alpha13 = 1,
  alpha14 = 1,
  alpha15 = 1,
  alpha16 = 1,
  alpha17 = 1,
  alpha18 = 1,
  alpha19 = 1,
  alpha20 = 1,
  alpha21 = 1,
  alpha22 = 1,
  alpha23 = 1,
  alpha24 = 1,
  alpha25 = 1,
  alpha26 = 1,
  alpha27 = 1,
  alpha28 = 1,
  alpha29 = 1,
  alpha30 = 1,
  alpha31 = 1,
  alpha32 = 1,
  alpha33 = 1,
  alpha34 = 1,
  alpha35 = 1,
  alpha36 = 1,
  alpha37 = 1,
  alpha38 = 1,
  alpha39 = 1,
  alpha40 = 1,
  iota1 = 0,
  iota2 = 0,
  iota3 = 0,
  iota4 = 0,
  iota5 = 0,
  iota6 = 0,
  iota7 = 0,
  iota8 = 0,
  iota9 = 0,
  iota10 = 0,
  iota11 = 0,
  iota12 = 0,
  iota13 = 0,
  iota14 = 0,
  iota15 = 0,
  iota16 = 0,
  iota17 = 0,
  iota18 = 0,
  iota19 = 0,
  iota20 = 0,
  iota21 = 0,
  iota22 = 0,
  iota23 = 0,
  iota24 = 0,
  iota25 = 0,
  iota26 = 0,
  iota27 = 0,
  iota28 = 0,
  iota29 = 0,
  iota30 = 0,
  iota31 = 0,
  iota32 = 0,
  iota33 = 0,
  iota34 = 0,
  iota35 = 0,
  iota36 = 0,
  iota37 = 0,
  iota38 = 0,
  iota39 = 0,
  iota40 = 0,
  c1 = 0.032121729,
  c2 = 0.032121729,
  c3 = 0.032121729,
  c4 = 0.032121729,
  c5 = 0.032121729,
  c6 = 0.032121729,
  c7 = 0.032121729,
  c8 = 0.032121729,
  c9 = 0.032121729,
  c10 = 0.032121729,
  c11 = 0.032121729,
  c12 = 0.032121729,
  c13 = 0.032121729,
  c14 = 0.032121729,
  c15 = 0.032121729,
  c16 = 0.032121729,
  c17 = 0.032121729,
  c18 = 0.032121729,
  c19 = 0.032121729,
  c20 = 0.032121729,
  c21 = 0.032121729,
  c22 = 0.032121729,
  c23 = 0.032121729,
  c24 = 0.032121729,
  c25 = 0.032121729,
  c26 = 0.032121729,
  c27 = 0.032121729,
  c28 = 0.032121729,
  c29 = 0.032121729,
  c30 = 0.032121729,
  c31 = 0.032121729,
  c32 = 0.032121729,
  c33 = 0.032121729,
  c34 = 0.032121729,
  c35 = 0.032121729,
  c36 = 0.032121729,
  c37 = 0.032121729,
  c38 = 0.032121729,
  c39 = 0.032121729,
  c40 = 0.032121729,
  a1 = 0.37107605,
  a2 = 0.37107605,
  a3 = 0.37107605,
  a4 = 0.37107605,
  a5 = 0.37107605,
  a6 = 0.37107605,
  a7 = 0.37107605,
  a8 = 0.37107605,
  a9 = 0.37107605,
  a10 = 0.37107605,
  a11 = 0.37107605,
  a12 = 0.37107605,
  a13 = 0.37107605,
  a14 = 0.37107605,
  a15 = 0.37107605,
  a16 = 0.37107605,
  a17 = 0.37107605,
  a18 = 0.37107605,
  a19 = 0.37107605,
  a20 = 0.37107605,
  a21 = 0.37107605,
  a22 = 0.37107605,
  a23 = 0.37107605,
  a24 = 0.37107605,
  a25 = 0.37107605,
  a26 = 0.37107605,
  a27 = 0.37107605,
  a28 = 0.37107605,
  a29 = 0.37107605,
  a30 = 0.37107605,
  a31 = 0.37107605,
  a32 = 0.37107605,
  a33 = 0.37107605,
  a34 = 0.37107605,
  a35 = 0.37107605,
  a36 = 0.37107605,
  a37 = 0.37107605,
  a38 = 0.37107605,
  a39 = 0.37107605,
  a40 = 0.37107605,
  rho1 = 0.11784376,
  rho2 = 0.11784376,
  rho3 = 0.11784376,
  rho4 = 0.11784376,
  rho5 = 0.11784376,
  rho6 = 0.11784376,
  rho7 = 0.11784376,
  rho8 = 0.11784376,
  rho9 = 0.11784376,
  rho10 = 0.11784376,
  rho11 = 0.11784376,
  rho12 = 0.11784376,
  rho13 = 0.11784376,
  rho14 = 0.11784376,
  rho15 = 0.11784376,
  rho16 = 0.11784376,
  rho17 = 0.11784376,
  rho18 = 0.11784376,
  rho19 = 0.11784376,
  rho20 = 0.11784376,
  rho21 = 0.11784376,
  rho22 = 0.11784376,
  rho23 = 0.11784376,
  rho24 = 0.11784376,
  rho25 = 0.11784376,
  rho26 = 0.11784376,
  rho27 = 0.11784376,
  rho28 = 0.11784376,
  rho29 = 0.11784376,
  rho30 = 0.11784376,
  rho31 = 0.11784376,
  rho32 = 0.11784376,
  rho33 = 0.11784376,
  rho34 = 0.11784376,
  rho35 = 0.11784376,
  rho36 = 0.11784376,
  rho37 = 0.11784376,
  rho38 = 0.11784376,
  rho39 = 0.11784376,
  rho40 = 0.11784376,
  gamma1 = 0.10496352,
  gamma2 = 0.10496352,
  gamma3 = 0.10496352,
  gamma4 = 0.10496352,
  gamma5 = 0.10496352,
  gamma6 = 0.10496352,
  gamma7 = 0.10496352,
  gamma8 = 0.10496352,
  gamma9 = 0.10496352,
  gamma10 = 0.10496352,
  gamma11 = 0.10496352,
  gamma12 = 0.10496352,
  gamma13 = 0.10496352,
  gamma14 = 0.10496352,
  gamma15 = 0.10496352,
  gamma16 = 0.10496352,
  gamma17 = 0.10496352,
  gamma18 = 0.10496352,
  gamma19 = 0.10496352,
  gamma20 = 0.10496352,
  gamma21 = 0.10496352,
  gamma22 = 0.10496352,
  gamma23 = 0.10496352,
  gamma24 = 0.10496352,
  gamma25 = 0.10496352,
  gamma26 = 0.10496352,
  gamma27 = 0.10496352,
  gamma28 = 0.10496352,
  gamma29 = 0.10496352,
  gamma30 = 0.10496352,
  gamma31 = 0.10496352,
  gamma32 = 0.10496352,
  gamma33 = 0.10496352,
  gamma34 = 0.10496352,
  gamma35 = 0.10496352,
  gamma36 = 0.10496352,
  gamma37 = 0.10496352,
  gamma38 = 0.10496352,
  gamma39 = 0.10496352,
  gamma40 = 0.10496352,
  delta1 = 0.0001923076923076923,
  delta2 = 0.0001923076923076923,
  delta3 = 0.0001923076923076923,
  delta4 = 0.0001923076923076923,
  delta5 = 0.0001923076923076923,
  delta6 = 0.0001923076923076923,
  delta7 = 0.0001923076923076923,
  delta8 = 0.0001923076923076923,
  delta9 = 0.0001923076923076923,
  delta10 = 0.0001923076923076923,
  delta11 = 0.0001923076923076923,
  delta12 = 0.0001923076923076923,
  delta13 = 0.0001923076923076923,
  delta14 = 0.0001923076923076923,
  delta15 = 0.0001923076923076923,
  delta16 = 0.0001923076923076923,
  delta17 = 0.0001923076923076923,
  delta18 = 0.0001923076923076923,
  delta19 = 0.0001923076923076923,
  delta20 = 0.0001923076923076923,
  delta21 = 0.0001923076923076923,
  delta22 = 0.0001923076923076923,
  delta23 = 0.0001923076923076923,
  delta24 = 0.0001923076923076923,
  delta25 = 0.0001923076923076923,
  delta26 = 0.0001923076923076923,
  delta27 = 0.0001923076923076923,
  delta28 = 0.0001923076923076923,
  delta29 = 0.0001923076923076923,
  delta30 = 0.0001923076923076923,
  delta31 = 0.0001923076923076923,
  delta32 = 0.0001923076923076923,
  delta33 = 0.0001923076923076923,
  delta34 = 0.0001923076923076923,
  delta35 = 0.0001923076923076923,
  delta36 = 0.0001923076923076923,
  delta37 = 0.0001923076923076923,
  delta38 = 0.0001923076923076923,
  delta39 = 0.0001923076923076923,
  delta40 = 0.0001923076923076923,
  sigma_xi1 = 0.1776823,
  sigma_xi2 = 0.1776823,
  sigma_xi3 = 0.1776823,
  sigma_xi4 = 0.1776823,
  sigma_xi5 = 0.1776823,
  sigma_xi6 = 0.1776823,
  sigma_xi7 = 0.1776823,
  sigma_xi8 = 0.1776823,
  sigma_xi9 = 0.1776823,
  sigma_xi10 = 0.1776823,
  sigma_xi11 = 0.1776823,
  sigma_xi12 = 0.1776823,
  sigma_xi13 = 0.1776823,
  sigma_xi14 = 0.1776823,
  sigma_xi15 = 0.1776823,
  sigma_xi16 = 0.1776823,
  sigma_xi17 = 0.1776823,
  sigma_xi18 = 0.1776823,
  sigma_xi19 = 0.1776823,
  sigma_xi20 = 0.1776823,
  sigma_xi21 = 0.1776823,
  sigma_xi22 = 0.1776823,
  sigma_xi23 = 0.1776823,
  sigma_xi24 = 0.1776823,
  sigma_xi25 = 0.1776823,
  sigma_xi26 = 0.1776823,
  sigma_xi27 = 0.1776823,
  sigma_xi28 = 0.1776823,
  sigma_xi29 = 0.1776823,
  sigma_xi30 = 0.1776823,
  sigma_xi31 = 0.1776823,
  sigma_xi32 = 0.1776823,
  sigma_xi33 = 0.1776823,
  sigma_xi34 = 0.1776823,
  sigma_xi35 = 0.1776823,
  sigma_xi36 = 0.1776823,
  sigma_xi37 = 0.1776823,
  sigma_xi38 = 0.1776823,
  sigma_xi39 = 0.1776823,
  sigma_xi40 = 0.1776823,
  psi1 = 0.20831605,
  psi2 = 0.20831605,
  psi3 = 0.20831605,
  psi4 = 0.20831605,
  psi5 = 0.20831605,
  psi6 = 0.20831605,
  psi7 = 0.20831605,
  psi8 = 0.20831605,
  psi9 = 0.20831605,
  psi10 = 0.20831605,
  psi11 = 0.20831605,
  psi12 = 0.20831605,
  psi13 = 0.20831605,
  psi14 = 0.20831605,
  psi15 = 0.20831605,
  psi16 = 0.20831605,
  psi17 = 0.20831605,
  psi18 = 0.20831605,
  psi19 = 0.20831605,
  psi20 = 0.20831605,
  psi21 = 0.20831605,
  psi22 = 0.20831605,
  psi23 = 0.20831605,
  psi24 = 0.20831605,
  psi25 = 0.20831605,
  psi26 = 0.20831605,
  psi27 = 0.20831605,
  psi28 = 0.20831605,
  psi29 = 0.20831605,
  psi30 = 0.20831605,
  psi31 = 0.20831605,
  psi32 = 0.20831605,
  psi33 = 0.20831605,
  psi34 = 0.20831605,
  psi35 = 0.20831605,
  psi36 = 0.20831605,
  psi37 = 0.20831605,
  psi38 = 0.20831605,
  psi39 = 0.20831605,
  psi40 = 0.20831605,
  g1 = 621.00315,
  g2 = 621.00315,
  g3 = 621.00315,
  g4 = 621.00315,
  g5 = 621.00315,
  g6 = 621.00315,
  g7 = 621.00315,
  g8 = 621.00315,
  g9 = 621.00315,
  g10 = 621.00315,
  g11 = 621.00315,
  g12 = 621.00315,
  g13 = 621.00315,
  g14 = 621.00315,
  g15 = 621.00315,
  g16 = 621.00315,
  g17 = 621.00315,
  g18 = 621.00315,
  g19 = 621.00315,
  g20 = 621.00315,
  g21 = 621.00315,
  g22 = 621.00315,
  g23 = 621.00315,
  g24 = 621.00315,
  g25 = 621.00315,
  g26 = 621.00315,
  g27 = 621.00315,
  g28 = 621.00315,
  g29 = 621.00315,
  g30 = 621.00315,
  g31 = 621.00315,
  g32 = 621.00315,
  g33 = 621.00315,
  g34 = 621.00315,
  g35 = 621.00315,
  g36 = 621.00315,
  g37 = 621.00315,
  g38 = 621.00315,
  g39 = 621.00315,
  g40 = 621.00315,
  betabar1  = 0.88894647,
  betabar2  = 0.59131747,
  betabar3  = 1.3571352,
  betabar4  = 0.99088466,
  betabar5  = 1.1142877,
  betabar6  = 1.073081,
  betabar7  = 1.5383922,
  betabar8  = 0.90597922,
  betabar9  = 0.82007521,
  betabar10 = 0.85654908,
  betabar11 = 1.0977113,
  betabar12 = 0.86888778,
  betabar13 = 1.1310883,
  betabar14 = 0.9737336,
  betabar15 = 0.94829279,
  betabar16 = 0.62169701,
  betabar17 = 0.94519132,
  betabar18 = 0.59131724,
  betabar19 = 0.5913173,
  betabar20 = 0.76468575,
  betabar21 = 1.188135,
  betabar22 = 0.96471751,
  betabar23 = 1.3417162,
  betabar24 = 1.2114047,
  betabar25 = 1.2141544,
  betabar26 = 1.4035884,
  betabar27 = 0.98535007,
  betabar28 = 0.78903711,
  betabar29 = 1.3568828,
  betabar30 = 0.92531413,
  betabar31 = 0.85022265,
  betabar32 = 1.1698065,
  betabar33 = 1.747846,
  betabar34 = 1.0385478,
  betabar35 = 0.94807833,
  betabar36 = 0.90443295,
  betabar37 = 1.564422,
  betabar38 = 1.3209716,
  betabar39 = 1.4434845,
  betabar40 = 0.89985961,
  S_01 = 0.075939171,
  S_02 = 0.045746908,
  S_03 = 0.072099209,
  S_04 = 0.093366429,
  S_05 = 0.12166313,
  S_06 = 0.082433015,
  S_07 = 0.076253399,
  S_08 = 0.073552363,
  S_09 = 0.091899686,
  S_010 = 0.11601865,
  S_011 = 0.11060591,
  S_012 = 0.086007304,
  S_013 = 0.090033337,
  S_014 = 0.068562545,
  S_015 = 0.08527638,
  S_016 = 0.099353261,
  S_017 = 0.071033828,
  S_018 = 0.057539228,
  S_019 = 0.0457469,
  S_020 = 0.056584992,
  S_021 = 0.066623405,
  S_022 = 0.071289554,
  S_023 = 0.10958349,
  S_024 = 0.057944749,
  S_025 = 0.098137014,
  S_026 = 0.083878987,
  S_027 = 0.073387422,
  S_028 = 0.065006606,
  S_029 = 0.076856337,
  S_030 = 0.089820623,
  S_031 = 0.078178115,
  S_032 = 0.11018048,
  S_033 = 0.1181622,
  S_034 = 0.098707311,
  S_035 = 0.09720967,
  S_036 = 0.076138817,
  S_037 = 0.066983677,
  S_038 = 0.081948154,
  S_039 = 0.083776928,
  S_040 = 0.063756049,
  E_01 = 6.5060573e-05,
  E_02 = 4.5087443e-05,
  E_03 = 5.2817129e-05,
  E_04 = 6.8182577e-05,
  E_05 = 6.4530977e-05,
  E_06 = 4.4786735e-05,
  E_07 = 7.5147931e-05,
  E_08 = 9.4057032e-05,
  E_09 = 4.7683639e-05,
  E_010 = 4.2368429e-05,
  E_011 = 6.2710962e-05,
  E_012 = 7.0720002e-05,
  E_013 = 6.5743807e-05,
  E_014 = 9.3865732e-05,
  E_015 = 9.4957613e-05,
  E_016 = 5.9195183e-05,
  E_017 = 4.9652379e-05,
  E_018 = 4.4339824e-05,
  E_019 = 5.5780311e-05,
  E_020 = 3.4462537e-05,
  E_021 = 6.6237786e-05,
  E_022 = 4.6263423e-05,
  E_023 = 4.987068e-05,
  E_024 = 4.3129025e-05,
  E_025 = 7.405701e-05,
  E_026 = 6.1690727e-05,
  E_027 = 7.0626847e-05,
  E_028 = 7.152609e-05,
  E_029 = 5.3734158e-05,
  E_030 = 6.6238754e-05,
  E_031 = 5.7956771e-05,
  E_032 = 6.3070715e-05,
  E_033 = 8.4001782e-05,
  E_034 = 8.3680527e-05,
  E_035 = 3.8972536e-05,
  E_036 = 5.995209e-05,
  E_037 = 5.8326987e-05,
  E_038 = 5.703792e-05,
  E_039 = 8.9273613e-05,
  E_040 = 6.1138249e-05,
  I_01 = 1.25675637e-04,
  I_02 = 1.28807835e-04,
  I_03 = 0.0025939241,
  I_04 = 0.0011983996,
  I_05 = 5.3031814e-05,
  I_06 = 0.00043914493,
  I_07 = 0.00035218208,
  I_08 = 0.0020560939,
  I_09 = 0.00058301724,
  I_010 = 0.00017986108,
  I_011 = 7.6727294e-05,
  I_012 = 6.1292536e-05,
  I_013 = 0.00021765193,
  I_014 = 0.00083288341,
  I_015 = 1.3892895e-08,
  I_016 = 0.0016802071,
  I_017 = 0.0013289195,
  I_018 = 0.00024760532,
  I_019 = 0.00020639459,
  I_020 = 0.0014469447,
  I_021 = 0.00066906348,
  I_022 = 0.00023718234,
  I_023 = 1.6463227e-08,
  I_024 = 0.0006781175,
  I_025 = 0.00091942732,
  I_026 = 0.00076855475,
  I_027 = 0.0029490807,
  I_028 = 0.0024151804,
  I_029 = 0.0013736223,
  I_030 = 0.0010850356,
  I_031 = 0.0030775415,
  I_032 = 0.0039083785,
  I_033 = 0.00043130366,
  I_034 = 0.0025644207,
  I_035 = 0.0023132868,
  I_036 = 0.00070547545,
  I_037 = 0.00088096713,
  I_038 = 0.00051698624,
  I_039 = 0.00079949765,
  I_040 = 0.00021292546,
  gaussianrho1  = 0.512295,
  gaussianrho2  = 0.5846346,
  gaussianrho3  = 0.73733903,
  gaussianrho4  = 0.64644552,
  gaussianrho5  = 0.68464784,
  gaussianrho6  = 0.62805587,
  gaussianrho7  = 0.59746622,
  gaussianrho8  = 0.62227199,
  gaussianrho9  = 0.57668756,
  gaussianrho10 = 0.68815189,
  gaussianrho11 = 0.75385802,
  gaussianrho12 = 0.63758144,
  gaussianrho13 = 0.65418166,
  gaussianrho14 = 0.59742807,
  gaussianrho15 = 0.58928316,
  gaussianrho16 = 0.7143576,
  gaussianrho17 = 0.64989739,
  gaussianrho18 = 0.52764322,
  gaussianrho19 = 0.51610781,
  gaussianrho20 = 0.61995668,
  gaussianrho21 = 0.63451302,
  gaussianrho22 = 0.64841047,
  gaussianrho23 = 0.66641999,
  gaussianrho24 = 0.58411126,
  gaussianrho25 = 0.74216196,
  gaussianrho26 = 0.65110744,
  gaussianrho27 = 0.57886699,
  gaussianrho28 = 0.6156806,
  gaussianrho29 = 0.51067983,
  gaussianrho30 = 0.47119888,
  gaussianrho31 = 0.70101722,
  gaussianrho32 = 0.59482185,
  gaussianrho33 = 0.76027942,
  gaussianrho34 = 0.66776662,
  gaussianrho35 = 0.52625039,
  gaussianrho36 = 0.60342687,
  gaussianrho37 = 0.68950329,
  gaussianrho38 = 0.5869341,
  gaussianrho39 = 0.70792855,
  gaussianrho40 = 0.57809921
)

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

measles_dir <- paste0(pomp_dir,"measles_more",i,"/")
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


m1 <- ibpf_mle_searches[[9]]

spatPomp_dir <- paste0(pomp_dir,"E_",11,"/")
if(!dir.exists(spatPomp_dir)) dir.create(spatPomp_dir)

stew(file=paste0(spatPomp_dir,"E11.rda"),seed=124,{
  cat(capture.output(sessionInfo()),
      file=paste0(spatPomp_dir,"sessionInfo.txt"),sep="\n")
  
  bpf_logLik_40 <- foreach(i = 1:20, .combine = c) %dopar% {
    logLik(bpfilter(m1, Np = 100000, block_size = 1))
  }
})


