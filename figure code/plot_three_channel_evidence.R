library(tidyverse)
library(patchwork)

options(stringsAsFactors = FALSE)

# =========================
# 1. 路径设置
# =========================
run_dir  <- "E:/Desktop/data_cell-level"
data_dir <- "E:/Desktop/data_cat(out_dir, "\n")
dir.exists(out_dir)cell-level"
out_dir  <- file.path(run_dir, "paper_figures_3_4_4")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

expr_file     <- file.path(data_dir, "testing_sample.csv")
label_file    <- file.path(data_dir, "testing_label.csv")
gene_file     <- file.path(run_dir, "final_gene_list.csv")
evidence_file <- file.path(run_dir, "test_evidence_merged.csv")

# =========================
# 2. 读取数据
# =========================
expr_mat <- as.matrix(read.csv(expr_file, header = FALSE, check.names = FALSE))
labels   <- read.csv(label_file, header = FALSE)[, 1]
genes    <- read.csv(gene_file, header = FALSE, stringsAsFactors = FALSE)[, 1]
evi_df   <- read.csv(evidence_file, check.names = FALSE)

colnames(expr_mat) <- genes

label_map <- c("healthy", "COVID-19", "fluA")
labels_chr <- label_map[labels + 1]

# =========================
# 3. 自动识别 evidence 列
# =========================
find_col <- function(nms, pattern_vec) {
  hit <- nms[Reduce(`|`, lapply(pattern_vec, function(p) grepl(p, nms, ignore.case = TRUE)))]
  if (length(hit) == 0) return(NA)
  hit[1]
}

nms <- colnames(evi_df)

healthy_col <- find_col(nms, c("healthy.*evidence", "evidence.*healthy"))
covid_col   <- find_col(nms, c("covid.*evidence", "evidence.*covid"))
flua_col    <- find_col(nms, c("flua.*evidence", "evidence.*flua", "influenza.*evidence"))

if (any(is.na(c(healthy_col, covid_col, flua_col)))) {
  stop("没有识别出 evidence 列名，请检查 test_evidence_merged.csv")
}

plot_df <- data.frame(
  label = labels_chr,
  healthy_evidence = evi_df[[healthy_col]],
  covid_evidence   = evi_df[[covid_col]],
  flua_evidence    = evi_df[[flua_col]]
)

# =========================
# 4. 定义代表性功能模块
# =========================
gene_sets <- list(
  COVID_cytokine_inflammation = c(
    "NFKBIA","CXCL8","S100A8","DUSP1","ZFP36","LITAF","IER2","KLF6","FPR1"
  ),
  COVID_myeloid_migration = c(
    "PLEK","S100A8","CXCL8","FPR1","LITAF","TREM1","TYROBP","MNDA","CD55"
  ),
  FluA_antigen_presentation = c(
    "HLA-A","HLA-B","HLA-C","B2M","ANXA1","EVI2B","PSCA"
  ),
  FluA_antiviral_effector = c(
    "TXNIP","NAMPT","FTH1","S100A9","HLA-A","HLA-B","HLA-C","B2M"
  )
)

# =========================
# 5. pathway score 计算
# 每个基因按列 z-score，再对模块求平均
# =========================
z_expr <- scale(expr_mat)

calc_module_score <- function(mat_z, gene_vec) {
  genes_use <- intersect(gene_vec, colnames(mat_z))
  if (length(genes_use) == 0) {
    return(rep(NA_real_, nrow(mat_z)))
  }
  rowMeans(mat_z[, genes_use, drop = FALSE], na.rm = TRUE)
}

for (nm in names(gene_sets)) {
  plot_df[[nm]] <- calc_module_score(z_expr, gene_sets[[nm]])
}

# =========================
# 6. 相关性计算函数
# =========================
safe_spearman <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]
  y <- y[ok]
  
  if (length(x) < 3) {
    return(list(rho = NA_real_, p = NA_real_))
  }
  
  ct <- suppressWarnings(cor.test(x, y, method = "spearman", exact = FALSE))
  list(rho = unname(ct$estimate), p = ct$p.value)
}

module_names <- names(gene_sets)
evidence_names <- c("healthy_evidence", "covid_evidence", "flua_evidence")

# =========================
# 7. 全体细胞总体相关性
# =========================
cor_grid <- expand.grid(
  module = module_names,
  evidence = evidence_names,
  stringsAsFactors = FALSE
)

cor_res <- purrr::pmap_dfr(
  cor_grid,
  function(module, evidence) {
    tmp <- safe_spearman(plot_df[[module]], plot_df[[evidence]])
    tibble::tibble(
      module = module,
      evidence = evidence,
      rho = tmp$rho,
      p_value = tmp$p
    )
  }
) %>%
  dplyr::mutate(
    p_adj = p.adjust(p_value, method = "BH")
  )

write.csv(
  cor_res,
  file.path(out_dir, "pathway_evidence_correlation_all_cells.csv"),
  row.names = FALSE
)

# =========================
# 8. 真实类别内部的 matched correlation
# =========================
within_list <- list()

# COVID-19 子集
covid_df <- plot_df %>% dplyr::filter(label == "COVID-19")
for (m in c("COVID_cytokine_inflammation", "COVID_myeloid_migration")) {
  tmp <- safe_spearman(covid_df[[m]], covid_df$covid_evidence)
  within_list[[length(within_list) + 1]] <- tibble::tibble(
    label_subset = "COVID-19",
    module = m,
    evidence = "covid_evidence",
    rho = tmp$rho,
    p_value = tmp$p
  )
}

# fluA 子集
flua_df <- plot_df %>% dplyr::filter(label == "fluA")
for (m in c("FluA_antigen_presentation", "FluA_antiviral_effector")) {
  tmp <- safe_spearman(flua_df[[m]], flua_df$flua_evidence)
  within_list[[length(within_list) + 1]] <- tibble::tibble(
    label_subset = "fluA",
    module = m,
    evidence = "flua_evidence",
    rho = tmp$rho,
    p_value = tmp$p
  )
}

within_res <- dplyr::bind_rows(within_list) %>%
  dplyr::mutate(
    p_adj = p.adjust(p_value, method = "BH")
  )

write.csv(
  within_res,
  file.path(out_dir, "pathway_evidence_correlation_within_class.csv"),
  row.names = FALSE
)

# =========================
# 9. 总体热图
# =========================
heat_df <- cor_res %>%
  dplyr::mutate(
    module = factor(module, levels = rev(module_names)),
    evidence = factor(evidence, levels = evidence_names),
    label_txt = ifelse(is.na(rho), "NA", sprintf("%.2f", rho))
  )

p_heat <- ggplot(heat_df, aes(x = evidence, y = module, fill = rho)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = label_txt), size = 4) +
  scale_fill_gradient2(
    low = "#2C7FB8",
    mid = "white",
    high = "#D7301F",
    midpoint = 0,
    limits = c(-1, 1),
    na.value = "grey90"
  ) +
  labs(
    title = "Pathway score vs evidence correlation",
    x = "Evidence channel",
    y = NULL,
    fill = "Spearman rho"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    panel.grid = element_blank(),
    axis.text = element_text(color = "black")
  )

ggsave(
  file.path(out_dir, "Fig6A_pathway_evidence_heatmap.png"),
  p_heat, width = 7.5, height = 5.5, dpi = 400, bg = "white"
)
ggsave(
  file.path(out_dir, "Fig6A_pathway_evidence_heatmap.pdf"),
  p_heat, width = 7.5, height = 5.5, bg = "white"
)

# =========================
# 10. matched module 散点图
# =========================
get_rho <- function(df, module_name) {
  x <- df %>%
    dplyr::filter(module == module_name) %>%
    dplyr::pull(rho)
  if (length(x) == 0) return(NA_real_)
  x[1]
}

make_scatter <- function(df, xvar, yvar, title_text, rho_txt) {
  ggplot(df, aes_string(x = xvar, y = yvar)) +
    geom_point(alpha = 0.35, size = 1.2, color = "#2C7FB8") +
    geom_smooth(method = "lm", se = FALSE, color = "#D7301F", linewidth = 0.9) +
    labs(
      title = title_text,
      x = xvar,
      y = yvar
    ) +
    annotate(
      "text",
      x = Inf, y = Inf,
      label = rho_txt,
      hjust = 1.05, vjust = 1.5,
      size = 4
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      panel.grid.major = element_line(color = "grey92", linewidth = 0.3),
      panel.grid.minor = element_blank()
    )
}

rho1 <- get_rho(within_res, "COVID_cytokine_inflammation")
rho2 <- get_rho(within_res, "COVID_myeloid_migration")
rho3 <- get_rho(within_res, "FluA_antigen_presentation")
rho4 <- get_rho(within_res, "FluA_antiviral_effector")

p1 <- make_scatter(
  covid_df,
  "COVID_cytokine_inflammation",
  "covid_evidence",
  "A. COVID cytokine/inflammation vs COVID evidence",
  sprintf("rho = %.2f", rho1)
)

p2 <- make_scatter(
  covid_df,
  "COVID_myeloid_migration",
  "covid_evidence",
  "B. COVID myeloid/migration vs COVID evidence",
  sprintf("rho = %.2f", rho2)
)

p3 <- make_scatter(
  flua_df,
  "FluA_antigen_presentation",
  "flua_evidence",
  "C. FluA antigen presentation vs FluA evidence",
  sprintf("rho = %.2f", rho3)
)

p4 <- make_scatter(
  flua_df,
  "FluA_antiviral_effector",
  "flua_evidence",
  "D. FluA antiviral effector vs FluA evidence",
  sprintf("rho = %.2f", rho4)
)

fig_scatter <- (p1 + p2) / (p3 + p4)

ggsave(
  file.path(out_dir, "Fig6B_pathway_evidence_scatter.png"),
  fig_scatter, width = 11.5, height = 9, dpi = 400, bg = "white"
)
ggsave(
  file.path(out_dir, "Fig6B_pathway_evidence_scatter.pdf"),
  fig_scatter, width = 11.5, height = 9, bg = "white"
)

# =========================
# 11. 导出作图数据
# =========================
write.csv(
  plot_df,
  file.path(out_dir, "pathway_scores_with_evidence_per_cell.csv"),
  row.names = FALSE
)

cat("Done!\n")
cat("Output directory:\n")
cat(out_dir, "\n")