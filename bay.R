library(bnlearn)
library(tidyverse)
library(GGally)
library(ggcorrplot)
library(viridis)
library(infotheo)
library(lmtest)
library(car)
library(Rgraphviz)


data <- read.csv("data/senproperties_all.csv") %>%
  mutate(across(everything(), as.numeric)) %>%
  select(-c(X, senID)) %>%
  mutate(s_familiarity = w_familiarity * sen_length) %>%
  mutate(s_suprisal = w_suprisal * sen_length) %>%
  select(-c(w_length, w_familiarity, w_suprisal, sen_length))

data <- na.omit(data)


# ---- Method 1: Gaussian Bayesian Networks for continuous data ----

ggpairs(data)
summary(data)
str(data)

d <- data %>%
  mutate(sen_edit_distance = log1p(sen_edit_distance),
         s_suprisal        = log1p(s_suprisal))

d_plot <- d %>%
  rename("n clause"      = sen_clauses,
         "edit distance" = sen_edit_distance,
         "max depth"     = sen_maxDepth,
         "familiarity"   = s_familiarity,
         "surprisal"     = s_suprisal)

ggpairs(d_plot)

d_plot %>%
  pivot_longer(everything(), names_to = "var", values_to = "val") %>%
  ggplot(aes(val)) +
  geom_histogram(bins = 30) +
  facet_wrap(~ var, scales = "free")

sapply(d, function(x) mean((x - mean(x))^3) / sd(x)^3)

ggpairs(d_plot, lower = list(continuous = "smooth"))

d.corr <- cor(d_plot)

ggcorrplot(d.corr,
           type = "upper", show.diag = TRUE, lab = TRUE,
           lab_size = 4, tl.srt = 45,
           colors = c("#440154", "#21908C", "#FDE725"),
           legend.title = "corr coefficient",
           ggtheme = ggplot2::theme_minimal()) +
  scale_fill_gradientn(colors = viridis::viridis(100),
                       limits = c(0, 1),
                       breaks = seq(0, 1, 0.2),
                       name = " ") +
  theme(panel.grid = element_blank())

sentence_props <- c("sen_clauses", "sen_maxDepth", "s_familiarity", "s_suprisal")
behavioral     <- c("sen_edit_distance")

bl <- expand.grid(from = behavioral,
                  to   = sentence_props,
                  stringsAsFactors = FALSE)
print(bl)


set.seed(42)

str_hc <- boot.strength(
  data           = d,
  R              = 500,
  m              = nrow(d),
  algorithm      = "hc",
  algorithm.args = list(score = "bic-g", blacklist = bl)
)

str_hc[order(-str_hc$strength, -str_hc$direction), ]

thr <- attr(str_hc, "threshold")
cat("Data-driven threshold:", thr, "\n")
avg_hc <- averaged.network(str_hc, threshold = thr)
print(avg_hc)
modelstring(avg_hc)

str_tabu <- boot.strength(
  data           = d,
  R              = 500,
  m              = nrow(d),
  algorithm      = "tabu",
  algorithm.args = list(score = "bic-g", blacklist = bl)
)
avg_tabu <- averaged.network(str_tabu, threshold = attr(str_tabu, "threshold"))
modelstring(avg_tabu)

str_mmhc <- boot.strength(
  data           = d,
  R              = 500,
  m              = nrow(d),
  algorithm      = "mmhc",
  algorithm.args = list(blacklist = bl)
)
avg_mmhc <- averaged.network(str_mmhc, threshold = attr(str_mmhc, "threshold"))
directed.arcs(avg_mmhc)
undirected.arcs(avg_mmhc)

avg_mmhc_ext <- cextend(avg_mmhc)
modelstring(avg_mmhc_ext)

strength.plot(avg_hc, str_hc, shape = "ellipse", threshold = thr)


avg_hc   <- averaged.network(str_hc,   threshold = 0.9)
avg_tabu <- averaged.network(str_tabu, threshold = 0.9)
avg_mmhc <- averaged.network(str_mmhc, threshold = 0.9)

modelstring(avg_hc)
modelstring(avg_tabu)

directed.arcs(avg_mmhc)
undirected.arcs(avg_mmhc)
avg_mmhc_ext <- cextend(avg_mmhc)
modelstring(avg_mmhc_ext)

parents(avg_hc,       "sen_edit_distance")
parents(avg_tabu,     "sen_edit_distance")
parents(avg_mmhc_ext, "sen_edit_distance")


fit <- bn.fit(avg_hc, data = d, method = "mle-g")
fit$sen_edit_distance

d_z   <- as.data.frame(scale(d))
fit_z <- bn.fit(avg_hc, data = d_z, method = "mle-g")
fit_z$sen_edit_distance

m <- lm(sen_edit_distance ~ sen_maxDepth + s_familiarity + s_suprisal, data = d)
summary(m)
confint(m)

avg_hc_85 <- averaged.network(str_hc, threshold = 0.85)
parents(avg_hc_85, "sen_edit_distance")
fit_85 <- bn.fit(avg_hc_85, data = d, method = "mle-g")
fit_85$sen_edit_distance

m_85 <- lm(sen_edit_distance ~ sen_clauses + sen_maxDepth + s_familiarity + s_suprisal,
           data = d)
summary(m_85)$coefficients


set.seed(42)

cv_primary <- bn.cv(d, bn = avg_hc, loss = "logl", k = 10, runs = 5)

empty_dag <- empty.graph(names(d))
cv_empty  <- bn.cv(d, bn = empty_dag, loss = "logl", k = 10, runs = 5)

full_dag <- empty.graph(names(d))
arcs(full_dag) <- matrix(c(
  "sen_clauses",   "sen_edit_distance",
  "sen_maxDepth",  "sen_edit_distance",
  "s_familiarity", "sen_edit_distance",
  "s_suprisal",    "sen_edit_distance",
  "sen_maxDepth",  "sen_clauses",
  "sen_maxDepth",  "s_familiarity",
  "sen_maxDepth",  "s_suprisal",
  "s_familiarity", "sen_clauses",
  "s_familiarity", "s_suprisal",
  "sen_clauses",   "s_suprisal"
), ncol = 2, byrow = TRUE)
cv_full <- bn.cv(d, bn = full_dag, loss = "logl", k = 10, runs = 5)

print(cv_primary)
print(cv_empty)
print(cv_full)

m <- lm(sen_edit_distance ~ sen_maxDepth + s_familiarity + s_suprisal, data = d)

par(mfrow = c(2, 2))
plot(m)
par(mfrow = c(1, 1))

shapiro.test(residuals(m))
bptest(m)
vif(m)

ci.test("sen_clauses", "sen_edit_distance", "sen_maxDepth", data = d, test = "cor")
ci.test("sen_clauses", "s_suprisal", c("sen_maxDepth", "s_familiarity"), data = d, test = "cor")

sim <- rbn(fit, n = 1000)

par(mfrow = c(1, 2))
hist(d$sen_edit_distance, main = "Observed", xlab = "log edit distance",
     breaks = 30, col = "steelblue", border = "white")
hist(sim$sen_edit_distance, main = "Simulated", xlab = "log edit distance",
     breaks = 30, col = "tomato", border = "white")
par(mfrow = c(1, 1))

rbind(observed  = sapply(d,   function(x) c(mean = mean(x), sd = sd(x))),
      simulated = sapply(sim, function(x) c(mean = mean(x), sd = sd(x))))

strength.plot(
  x         = avg_hc,
  strength  = str_hc,
  threshold = 0.90,
  shape     = "ellipse",
  layout    = "dot",
  main      = "Bayesian network: scanpath deviation"
)


# ---- Method 2: Discretized data with Hill-Climbing structure learning ----

summary(data)

df <- data

par(mfrow = c(2, 3))
hist(df$sen_clauses,       main = "sen_clauses",       xlab = "Value", col = "skyblue",        border = "blue")
hist(df$sen_edit_distance, main = "sen_edit_distance", xlab = "Value", col = "lightgreen",     border = "darkgreen")
hist(df$sen_maxDepth,      main = "sen_maxDepth",      xlab = "Value", col = "lightcoral",     border = "red")
hist(df$s_familiarity,     main = "s_familiarity",     xlab = "Value", col = "lightgoldenrod", border = "goldenrod")
hist(df$s_suprisal,        main = "s_suprisal",        xlab = "Value", col = "lightpink",      border = "darkred")
par(mfrow = c(1, 1))

par(mfrow = c(2, 3))
plot(density(df$sen_clauses),       main = "sen_clauses",       col = "red")
plot(density(df$sen_edit_distance), main = "sen_edit_distance", col = "red")
plot(density(df$sen_maxDepth),      main = "sen_maxDepth",      col = "red")
plot(density(df$s_familiarity),     main = "s_familiarity",     col = "red")
plot(density(df$s_suprisal),        main = "s_suprisal",        col = "red")
par(mfrow = c(1, 1))


data <- as.data.frame(data)
n_bins <- 4

discretized_data <- as.data.frame(lapply(df, function(col) {
  cut(col, breaks = n_bins, labels = FALSE, include.lowest = TRUE)
}))
discretized_data[] <- lapply(discretized_data, as.factor)


bn_structure <- empty.graph(names(discretized_data))

arcs(bn_structure) <- matrix(c(
  "s_familiarity", "sen_edit_distance",
  "sen_clauses",   "sen_edit_distance",
  "s_suprisal",    "sen_edit_distance",
  "sen_clauses",   "sen_maxDepth",
  "sen_maxDepth",  "sen_edit_distance"
), ncol = 2, byrow = TRUE)

graphviz.plot(bn_structure)
bn_structure


bn_structure <- hc(discretized_data)
print(bn_structure)

graphviz.plot(bn_structure,
              shape = "ellipse",
              main  = "Customized Bayesian Network")

fitted_bn <- bn.fit(bn_structure, discretized_data)
fitted_bn

arc_strengths <- arc.strength(bn_structure, discretized_data)
strength.plot(bn_structure, arc_strengths)


mi_maxDepth    <- mutinformation(discretized_data$sen_edit_distance, discretized_data$sen_maxDepth)
mi_familiarity <- mutinformation(discretized_data$sen_edit_distance, discretized_data$s_familiarity)
mi_suprisal    <- mutinformation(discretized_data$sen_edit_distance, discretized_data$s_suprisal)

print(paste("Mutual Information with sen_maxDepth:",  mi_maxDepth))
print(paste("Mutual Information with s_familiarity:", mi_familiarity))
print(paste("Mutual Information with s_suprisal:",    mi_suprisal))


ci_test_maxDepth    <- ci.test(x = discretized_data$sen_edit_distance, y = discretized_data$sen_maxDepth,  test = "mi")
ci_test_familiarity <- ci.test(x = discretized_data$sen_edit_distance, y = discretized_data$s_familiarity, test = "mi")
ci_test_suprisal    <- ci.test(x = discretized_data$sen_edit_distance, y = discretized_data$s_suprisal,    test = "mi")

print(ci_test_maxDepth)
print(ci_test_familiarity)
print(ci_test_suprisal)