# =============================================================================
# Stylized fact validation: Phillips Curve and Okun's Law
#
# Phillips: inflation_it   = a + b1 * unemployment_it + city_i + e_it
# Okun:     gdp_growth_it  = a + b2 * unemp_diff_it    + city_i + e_it
#
# Adapted from the co-author's Stylized_facts_test(1).R, which is the script
# that produced Tables 1 and 6 of the submitted paper. Changes, all mechanical:
#   * the hardcoded Windows setwd() is gone; input files are arguments
#   * multiple input files are concatenated (the corrected batch arrived as the
#     25-city pack plus a separate Rio de Janeiro run)
#   * outputs are written next to this script with a batch-tagged name, so the
#     submitted and corrected estimates can sit side by side
#
# The estimation itself is untouched: same burn-in, same specs, same order.
#
# Usage:
#   Rscript analysis/validation/stylized_facts.R <tag> <file1.csv> [file2.csv ...]
# =============================================================================

library(data.table)
library(fixest)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) stop("usage: stylized_facts.R <tag> <stats.csv> [...]")
tag <- args[1]
files <- args[-1]

OUT_DIR <- file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE),
                                                      value = TRUE)[1])))
if (is.na(OUT_DIR) || OUT_DIR == "") OUT_DIR <- "."

# -
# 1. Load, sort, and cut the burn-in before touching anything else.
#    The first ~18 months are a transient - unemployment falls from ~11.8%
#    to ~2% as the model settles into its steady state. Leave it in and it
#    swamps the real correlations, to the point of flipping the sign of the
#    Phillips curve depending on spec.
# -
BURN_IN <- 18  # months dropped from the start of every run

cat("Loading:", paste(basename(files), collapse = ", "), "\n")
df <- rbindlist(lapply(files, fread), fill = TRUE)
df[, month := as.IDate(month)]
setorder(df, simulation_id, month)

cat("  ", nrow(df), "rows,", uniqueN(df$simulation_id), "sims,",
    uniqueN(df$processing_acps), "cities\n")

df[, t_index := seq_len(.N), by = simulation_id]
df <- df[t_index > BURN_IN]

df[, unemp_diff := unemployment - shift(unemployment, 1), by = simulation_id]

df[, city := processing_acps]
df[, scenario_cell := paste(interest_housing, policy_melhorias, sep = "_")]

# -
# 2. Phillips curve, full panel, post-burn-in, city FE, clustered SE
# -
m1 <- feols(inflation ~ unemployment | city, data = df, cluster = ~city)

# -
# 3. Okun's law, same panel
# -
m2 <- feols(gdp_growth_rate ~ unemp_diff | city, data = df, cluster = ~city)

# -
# 3b. Scenario FE on top of city FE. City FE only soaks up cross-city
#     differences, not cross-scenario ones; scenario FE forces beta to come from
#     variation WITHIN a given (interest_housing, policy_melhorias) cell.
# -
m2b <- feols(inflation ~ unemployment | city + interest_housing^policy_melhorias,
             data = df, cluster = ~city)
m2c <- feols(gdp_growth_rate ~ unemp_diff | city + interest_housing^policy_melhorias,
             data = df, cluster = ~city)

# -
# 4. Single-cell benchmark: media/False, full horizon, post-burn-in
# -
baseline <- df[interest_housing == "media" & policy_melhorias == FALSE]
baseline[, unemp_lag := shift(unemployment, 1), by = simulation_id]

m3 <- feols(inflation ~ unemployment | city, data = baseline, cluster = ~city)
m4 <- feols(inflation ~ unemp_lag | city, data = baseline, cluster = ~city)
m5 <- feols(gdp_growth_rate ~ unemp_diff | city, data = baseline, cluster = ~city)

for (nm in c("m1", "m2", "m2b", "m2c", "m3", "m4", "m5")) {
  cat("\n=== ", nm, " ===\n", sep = "")
  print(summary(get(nm)))
}

# -
# 4b. Slope heterogeneity: one slope PER scenario cell rather than differences
#     from a reference cell, since scenario_cell is absorbed on the FE side.
# -
m6 <- feols(inflation ~ unemployment:scenario_cell | city + scenario_cell,
            data = df, cluster = ~city)
m7 <- feols(gdp_growth_rate ~ unemp_diff:scenario_cell | city + scenario_cell,
            data = df, cluster = ~city)

# -
# 6. Results table: the seven FE specs (m1-m5 plus m2b/m2c), in the row order
#    the paper's Table 1 prints them.
# -
results <- data.table(
  spec = c("Phillips: full panel, contemp.",
           "Phillips: full panel, city FE + scenario FE",
           "Phillips: media/False, full horizon, contemp.",
           "Phillips: media/False, full horizon, lagged",
           "Okun: full panel",
           "Okun: full panel, city FE + scenario FE",
           "Okun: media/False, full horizon"),
  beta = c(coef(m1)["unemployment"], coef(m2b)["unemployment"],
           coef(m3)["unemployment"], coef(m4)["unemp_lag"],
           coef(m2)["unemp_diff"], coef(m2c)["unemp_diff"],
           coef(m5)["unemp_diff"]),
  std_error = c(se(m1)["unemployment"], se(m2b)["unemployment"],
                se(m3)["unemployment"], se(m4)["unemp_lag"],
                se(m2)["unemp_diff"], se(m2c)["unemp_diff"],
                se(m5)["unemp_diff"]),
  p_value = c(pvalue(m1)["unemployment"], pvalue(m2b)["unemployment"],
              pvalue(m3)["unemployment"], pvalue(m4)["unemp_lag"],
              pvalue(m2)["unemp_diff"], pvalue(m2c)["unemp_diff"],
              pvalue(m5)["unemp_diff"])
)
cat("\n=== results ===\n")
print(results)
fwrite(results, file.path(OUT_DIR, paste0("stylized_facts_results_", tag, ".csv")))

# -
# 7. Tidy export of the slope-heterogeneity tests (m6/m7), long format:
#    one row per (test, scenario_cell).
# -
tidy_by_cell <- function(model, test_label, prefix) {
  cf <- coef(model); s <- se(model); pv <- pvalue(model)
  data.table(test = test_label,
             scenario_cell = sub(prefix, "", names(cf)),
             beta = as.numeric(cf), std_error = as.numeric(s),
             p_value = as.numeric(pv))
}

results_by_cell <- rbind(
  tidy_by_cell(m6, "Phillips", "unemployment:scenario_cell"),
  tidy_by_cell(m7, "Okun",     "unemp_diff:scenario_cell")
)
setorder(results_by_cell, test, scenario_cell)
cat("\n=== results by cell ===\n")
print(results_by_cell)
fwrite(results_by_cell,
       file.path(OUT_DIR, paste0("stylized_facts_slope_by_cell_", tag, ".csv")))

cat("\nWrote *_", tag, ".csv to ", OUT_DIR, "\n", sep = "")
