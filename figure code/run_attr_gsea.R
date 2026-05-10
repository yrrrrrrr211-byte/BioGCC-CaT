library(tidyverse)
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(patchwork)

options(stringsAsFactors = FALSE)
options(timeout = 300)

# =========================
# 1. 路径设置
# =========================
run_dir  <- "E:/Desktop"
attr_dir <- file.path(run_dir, "holdout_attribution")
out_dir  <- file.path(run_dir, "paper_figures_3_4_3")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

covid_file <- file.path(attr_dir, "heldout_covid_gene_attr.csv")
flua_file  <- file.path(attr_dir, "heldout_flua_gene_attr.csv")

TOP_SHOW <- 12

# =========================
# 2. 工具函数
# =========================
read_attr_table <- function(fp) {
  df <- read.csv(fp, check.names = FALSE)
  stopifnot(all(c("gene", "attr_mean_signed") %in% colnames(df)))
  df$gene <- as.character(df$gene)
  df <- dplyr::distinct(df, gene, .keep_all = TRUE)
  return(df)
}

make_ranked_list <- function(df, drop_ribo = FALSE) {
  x <- df %>%
    dplyr::select(gene, attr_mean_signed) %>%
    dplyr::distinct(gene, .keep_all = TRUE)
  
  if (drop_ribo) {
    x <- x %>%
      dplyr::filter(!grepl("^RPL|^RPS", gene))
  }
  
  mapped <- suppressWarnings(
    bitr(
      x$gene,
      fromType = "SYMBOL",
      toType = "ENTREZID",
      OrgDb = org.Hs.eg.db
    )
  )
  
  if (is.null(mapped) || nrow(mapped) == 0) {
    stop("基因 SYMBOL -> ENTREZID 映射失败，无法构建 ranked list")
  }
  
  mapped <- mapped %>%
    dplyr::distinct(SYMBOL, .keep_all = TRUE)
  
  x2 <- x %>%
    dplyr::inner_join(mapped, by = c("gene" = "SYMBOL")) %>%
    dplyr::distinct(ENTREZID, .keep_all = TRUE)
  
  gene_list <- x2$attr_mean_signed
  names(gene_list) <- x2$ENTREZID
  gene_list <- sort(gene_list, decreasing = TRUE)
  
  return(gene_list)
}

run_gsea_go <- function(gene_list) {
  if (length(gene_list) < 50) return(NULL)
  
  suppressWarnings(
    gseGO(
      geneList      = gene_list,
      OrgDb         = org.Hs.eg.db,
      ont           = "BP",
      keyType       = "ENTREZID",
      minGSSize     = 10,
      maxGSSize     = 500,
      pvalueCutoff  = 0.05,
      pAdjustMethod = "BH",
      verbose       = FALSE
    )
  )
}

run_gsea_kegg <- function(gene_list) {
  if (length(gene_list) < 50) return(NULL)
  
  suppressWarnings(
    gseKEGG(
      geneList      = gene_list,
      organism      = "hsa",
      minGSSize     = 10,
      pvalueCutoff  = 0.05,
      pAdjustMethod = "BH",
      verbose       = FALSE
    )
  )
}

save_gsea_result <- function(obj, prefix, out_dir) {
  if (is.null(obj)) return(NULL)
  df <- as.data.frame(obj)
  if (nrow(df) == 0) return(NULL)
  write.csv(df, file.path(out_dir, paste0(prefix, ".csv")), row.names = FALSE)
  return(df)
}

plot_gsea_dot <- function(obj, title_text, out_file, top_n = 12) {
  if (is.null(obj)) return(NULL)
  df <- as.data.frame(obj)
  if (nrow(df) == 0) return(NULL)
  
  n_show <- min(top_n, nrow(df))
  
  p <- enrichplot::dotplot(obj, showCategory = n_show, split = ".sign") +
    ggplot2::facet_grid(. ~ .sign) +
    ggplot2::ggtitle(title_text) +
    ggplot2::theme_bw(base_size = 12) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
      panel.grid.major = ggplot2::element_line(color = "grey90", linewidth = 0.3),
      panel.grid.minor = ggplot2::element_blank()
    )
  
  ggplot2::ggsave(out_file, p, width = 10, height = 6, dpi = 400, bg = "white")
  return(p)
}

# =========================
# 3. 读取 attribution 数据
# =========================
covid_df <- read_attr_table(covid_file)
flua_df  <- read_attr_table(flua_file)

# =========================
# 4. 构建 ranked list
# =========================
covid_ranked <- make_ranked_list(covid_df, drop_ribo = FALSE)
covid_ranked_no_ribo <- make_ranked_list(covid_df, drop_ribo = TRUE)
flua_ranked  <- make_ranked_list(flua_df, drop_ribo = FALSE)

write.csv(
  data.frame(ENTREZID = names(covid_ranked), score = as.numeric(covid_ranked)),
  file.path(out_dir, "covid_ranked_list.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(ENTREZID = names(covid_ranked_no_ribo), score = as.numeric(covid_ranked_no_ribo)),
  file.path(out_dir, "covid_ranked_no_ribo_list.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(ENTREZID = names(flua_ranked), score = as.numeric(flua_ranked)),
  file.path(out_dir, "flua_ranked_list.csv"),
  row.names = FALSE
)

# =========================
# 5. 运行 GSEA
# =========================
covid_go <- run_gsea_go(covid_ranked)
covid_no_ribo_go <- run_gsea_go(covid_ranked_no_ribo)
flua_go <- run_gsea_go(flua_ranked)

covid_kegg <- run_gsea_kegg(covid_ranked)
covid_no_ribo_kegg <- run_gsea_kegg(covid_ranked_no_ribo)
flua_kegg <- run_gsea_kegg(flua_ranked)

save_gsea_result(covid_go, "COVID_gseGO_BP", out_dir)
save_gsea_result(covid_no_ribo_go, "COVID_no_ribo_gseGO_BP", out_dir)
save_gsea_result(flua_go, "FluA_gseGO_BP", out_dir)

save_gsea_result(covid_kegg, "COVID_gseKEGG", out_dir)
save_gsea_result(covid_no_ribo_kegg, "COVID_no_ribo_gseKEGG", out_dir)
save_gsea_result(flua_kegg, "FluA_gseKEGG", out_dir)

# =========================
# 6. 画图
# =========================
plot_gsea_dot(
  covid_no_ribo_go,
  "COVID no-ribo GSEA (GO BP)",
  file.path(out_dir, "COVID_no_ribo_gseGO_dotplot.png"),
  top_n = TOP_SHOW
)

plot_gsea_dot(
  flua_go,
  "FluA GSEA (GO BP)",
  file.path(out_dir, "FluA_gseGO_dotplot.png"),
  top_n = TOP_SHOW
)

plot_gsea_dot(
  covid_no_ribo_kegg,
  "COVID no-ribo GSEA (KEGG)",
  file.path(out_dir, "COVID_no_ribo_gseKEGG_dotplot.png"),
  top_n = TOP_SHOW
)

plot_gsea_dot(
  flua_kegg,
  "FluA GSEA (KEGG)",
  file.path(out_dir, "FluA_gseKEGG_dotplot.png"),
  top_n = TOP_SHOW
)

cat("Done!\n")
cat("Output directory:\n", out_dir, "\n")