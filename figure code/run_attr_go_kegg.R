library(tidyverse)
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(patchwork)

options(stringsAsFactors = FALSE)
options(timeout = 300)

# =========================================================
# 1. 路径设置
# =========================================================
run_dir  <- "E:/Desktop"
attr_dir <- file.path(run_dir, "holdout_attribution")
out_dir  <- file.path(run_dir, "paper_figures_3_4_2")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

covid_file <- file.path(attr_dir, "heldout_covid_gene_attr.csv")
flua_file  <- file.path(attr_dir, "heldout_flua_gene_attr.csv")

TOP_N <- 300
P_CUTOFF <- 0.05
Q_CUTOFF <- 0.20

# =========================================================
# 2. 工具函数
# =========================================================
read_attr_table <- function(fp) {
  df <- read.csv(fp, check.names = FALSE)
  stopifnot(all(c("gene", "attr_mean_signed") %in% colnames(df)))
  df$gene <- as.character(df$gene)
  df <- df %>% distinct(gene, .keep_all = TRUE)
  return(df)
}

pick_top_positive <- function(df, top_n = 300) {
  df %>%
    arrange(desc(attr_mean_signed)) %>%
    slice_head(n = top_n) %>%
    pull(gene) %>%
    unique()
}

drop_ribo <- function(genes) {
  genes[!grepl("^RPL|^RPS", genes, ignore.case = FALSE)]
}

map_symbol_to_entrez <- function(genes) {
  genes <- unique(genes)
  mapped <- suppressWarnings(
    bitr(
      genes,
      fromType = "SYMBOL",
      toType = "ENTREZID",
      OrgDb = org.Hs.eg.db
    )
  )
  mapped <- mapped %>% distinct(SYMBOL, .keep_all = TRUE)
  return(mapped)
}

build_universe_entrez <- function(all_genes) {
  universe_map <- map_symbol_to_entrez(all_genes)
  universe_entrez <- unique(universe_map$ENTREZID)
  return(list(map = universe_map, entrez = universe_entrez))
}

run_go_bp <- function(entrez_ids, universe_entrez) {
  if (length(entrez_ids) < 10) return(NULL)
  suppressWarnings(
    enrichGO(
      gene          = entrez_ids,
      universe      = universe_entrez,
      OrgDb         = org.Hs.eg.db,
      keyType       = "ENTREZID",
      ont           = "BP",
      pAdjustMethod = "BH",
      pvalueCutoff  = P_CUTOFF,
      qvalueCutoff  = Q_CUTOFF,
      readable      = TRUE
    )
  )
}

run_kegg <- function(entrez_ids, universe_entrez) {
  if (length(entrez_ids) < 10) return(NULL)
  suppressWarnings(
    enrichKEGG(
      gene          = entrez_ids,
      organism      = "hsa",
      universe      = universe_entrez,
      pAdjustMethod = "BH",
      pvalueCutoff  = P_CUTOFF,
      qvalueCutoff  = Q_CUTOFF
    )
  )
}

save_enrich_result <- function(obj, prefix, out_dir) {
  if (is.null(obj)) return(NULL)
  df <- as.data.frame(obj)
  if (nrow(df) == 0) return(NULL)
  write.csv(df, file.path(out_dir, paste0(prefix, ".csv")), row.names = FALSE)
  return(df)
}

plot_dot_safe <- function(obj, title_text, out_file, top_n = 12, width = 8, height = 6) {
  if (is.null(obj)) return(NULL)
  df <- as.data.frame(obj)
  if (nrow(df) == 0) return(NULL)
  
  n_show <- min(top_n, nrow(df))
  p <- dotplot(obj, showCategory = n_show, font.size = 11) +
    ggtitle(title_text) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
      panel.grid.minor = element_blank()
    )
  
  ggsave(out_file, p, width = width, height = height, dpi = 400, bg = "white")
  return(p)
}

analyze_one_set <- function(df, set_name, genes_for_test, universe_entrez, out_dir) {
  mapped <- map_symbol_to_entrez(genes_for_test)
  test_entrez <- unique(mapped$ENTREZID)
  
  write.csv(
    data.frame(gene = genes_for_test),
    file.path(out_dir, paste0(set_name, "_input_genes.csv")),
    row.names = FALSE
  )
  write.csv(
    mapped,
    file.path(out_dir, paste0(set_name, "_mapped_entrez.csv")),
    row.names = FALSE
  )
  
  ego <- run_go_bp(test_entrez, universe_entrez)
  ekk <- run_kegg(test_entrez, universe_entrez)
  
  save_enrich_result(ego, paste0(set_name, "_GO_BP"), out_dir)
  save_enrich_result(ekk, paste0(set_name, "_KEGG"), out_dir)
  
  plot_dot_safe(
    ego,
    paste0(set_name, " GO Biological Process"),
    file.path(out_dir, paste0(set_name, "_GO_BP_dotplot.png"))
  )
  plot_dot_safe(
    ekk,
    paste0(set_name, " KEGG pathways"),
    file.path(out_dir, paste0(set_name, "_KEGG_dotplot.png"))
  )
  
  return(list(go = ego, kegg = ekk))
}

# =========================================================
# 3. 读 attribution 数据
# =========================================================
covid_df <- read_attr_table(covid_file)
flua_df  <- read_attr_table(flua_file)

# universe 用完整可见基因空间
all_genes <- union(covid_df$gene, flua_df$gene)
universe_obj <- build_universe_entrez(all_genes)
universe_entrez <- universe_obj$entrez

write.csv(
  data.frame(gene = all_genes),
  file.path(out_dir, "universe_genes_symbol.csv"),
  row.names = FALSE
)
write.csv(
  universe_obj$map,
  file.path(out_dir, "universe_genes_mapped_entrez.csv"),
  row.names = FALSE
)

# =========================================================
# 4. 取 top positive genes
# =========================================================
covid_top <- pick_top_positive(covid_df, TOP_N)
covid_top_no_ribo <- drop_ribo(covid_top)
flua_top <- pick_top_positive(flua_df, TOP_N)

# =========================================================
# 5. 分析
# =========================================================
res_covid <- analyze_one_set(
  df = covid_df,
  set_name = "COVID_top300",
  genes_for_test = covid_top,
  universe_entrez = universe_entrez,
  out_dir = out_dir
)

res_covid_noribo <- analyze_one_set(
  df = covid_df,
  set_name = "COVID_top300_no_ribo",
  genes_for_test = covid_top_no_ribo,
  universe_entrez = universe_entrez,
  out_dir = out_dir
)

res_flua <- analyze_one_set(
  df = flua_df,
  set_name = "FluA_top300",
  genes_for_test = flua_top,
  universe_entrez = universe_entrez,
  out_dir = out_dir
)

# =========================================================
# 6. 额外拼一个 GO 总图（可选）
# =========================================================
go_plot_files <- c(
  file.path(out_dir, "COVID_top300_no_ribo_GO_BP_dotplot.png"),
  file.path(out_dir, "FluA_top300_GO_BP_dotplot.png")
)

cat("Done!\n")
cat("Output directory:\n", out_dir, "\n")