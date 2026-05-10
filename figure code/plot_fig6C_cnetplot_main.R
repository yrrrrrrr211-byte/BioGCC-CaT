suppressPackageStartupMessages({
  library(tidyverse)
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(enrichplot)
  library(ggplot2)
  library(patchwork)
})

# =========================
# 0. 输入目录
# =========================
in_dir <- "E:/Desktop/cell level/subtype_attr_gxi"
gene_universe_file <- "E:/Desktop/cell level/rna_name.csv"

# 输出目录 = 输入目录
out_dir <- in_dir

# =========================
# 1. 配置
# =========================
TOP_N_GENES <- 150
SHOW_CATEGORY <- 5
PADJ_CUTOFF <- 0.20

task_info <- tibble::tibble(
  task = c("G5c_naive_fluA", "unk_epi_fluA"),
  panel = c("A", "B"),
  subtype = c("G5c_naive", "unk_epi"),
  disease = c("FluA", "FluA"),
  subtitle_color = c("#148A8A", "#148A8A")
)

REMOVE_DISPLAY_ARTIFACTS <- TRUE
bad_pattern <- "^(RPL|RPS|MRPL|MRPS|MT-|MTRNR|MT-RNR|HBB|HBA)"

# =========================
# 2. 背景基因 universe
# =========================
universe_symbols <- read.csv(
  gene_universe_file,
  header = FALSE,
  stringsAsFactors = FALSE
)[, 1] %>%
  as.character() %>%
  unique()

if (REMOVE_DISPLAY_ARTIFACTS) {
  universe_symbols <- universe_symbols[!grepl(bad_pattern, universe_symbols, ignore.case = TRUE)]
}

message("Mapping universe ...")
universe_map <- clusterProfiler::bitr(
  universe_symbols,
  fromType = "SYMBOL",
  toType   = "ENTREZID",
  OrgDb    = org.Hs.eg.db
)

universe_entrez <- unique(universe_map$ENTREZID)

# =========================
# 3. 主题
# =========================
theme_cnet <- function(base_size = 11) {
  theme_void(base_size = base_size) +
    theme(
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.title       = element_text(size = base_size + 1.4, face = "bold", hjust = 0, colour = "#111111"),
      plot.subtitle    = element_text(size = base_size - 0.1, face = "bold", hjust = 0),
      legend.title     = element_text(size = base_size, face = "bold"),
      legend.text      = element_text(size = base_size - 0.2),
      plot.margin      = margin(8, 8, 8, 8)
    )
}

# =========================
# 4. 稳健版 cnetplot 封装
# =========================
safe_cnetplot <- function(ego_obj, show_n, gene_fc = NULL) {
  # 方案1：带 foldChange + node_label
  p <- tryCatch(
    enrichplot::cnetplot(
      ego_obj,
      showCategory = show_n,
      foldChange = gene_fc,
      node_label = "all"
    ),
    error = function(e) NULL
  )
  if (!is.null(p)) return(p)
  
  # 方案2：带 foldChange
  p <- tryCatch(
    enrichplot::cnetplot(
      ego_obj,
      showCategory = show_n,
      foldChange = gene_fc
    ),
    error = function(e) NULL
  )
  if (!is.null(p)) return(p)
  
  # 方案3：最保守
  p <- tryCatch(
    enrichplot::cnetplot(
      ego_obj,
      showCategory = show_n
    ),
    error = function(e) NULL
  )
  return(p)
}

# =========================
# 5. 逐 subtype 跑 GO
# =========================
plot_list <- list()
res_export <- list()

for (i in seq_len(nrow(task_info))) {
  ti <- task_info[i, ]
  f_in <- file.path(in_dir, paste0(ti$task, "_ranked_genes.csv"))
  df <- read.csv(f_in, stringsAsFactors = FALSE)
  
  df <- df %>%
    dplyr::filter(mean_attr > 0) %>%
    dplyr::arrange(dplyr::desc(mean_attr))
  
  if (REMOVE_DISPLAY_ARTIFACTS) {
    df <- df %>%
      dplyr::filter(!grepl(bad_pattern, gene, ignore.case = TRUE))
  }
  
  df <- df %>% dplyr::slice_head(n = TOP_N_GENES)
  
  gene_symbols <- unique(df$gene)
  
  gene_map <- clusterProfiler::bitr(
    gene_symbols,
    fromType = "SYMBOL",
    toType   = "ENTREZID",
    OrgDb    = org.Hs.eg.db
  )
  
  gene_entrez <- unique(gene_map$ENTREZID)
  
  if (length(gene_entrez) < 5) {
    message("Too few mapped genes for: ", ti$task)
    next
  }
  
  message("Running enrichGO for ", ti$task, " ...")
  ego <- clusterProfiler::enrichGO(
    gene          = gene_entrez,
    universe      = universe_entrez,
    OrgDb         = org.Hs.eg.db,
    keyType       = "ENTREZID",
    ont           = "BP",
    pAdjustMethod = "BH",
    pvalueCutoff  = 1,
    qvalueCutoff  = 1,
    readable      = TRUE
  )
  
  if (is.null(ego) || nrow(as.data.frame(ego)) == 0) {
    message("No GO results for: ", ti$task)
    next
  }
  
  ego_df <- as.data.frame(ego) %>%
    dplyr::filter(!is.na(p.adjust)) %>%
    dplyr::filter(p.adjust <= PADJ_CUTOFF)
  
  if (nrow(ego_df) == 0) {
    message("No significant GO terms for: ", ti$task)
    next
  }
  
  # 尝试去冗余；失败就直接用原结果
  ego_sig <- ego
  ego_sig@result <- ego_df
  
  ego_simple <- tryCatch(
    clusterProfiler::simplify(ego_sig, cutoff = 0.55, by = "p.adjust", select_fun = min),
    error = function(e) ego_sig
  )
  
  ego_simple_df <- as.data.frame(ego_simple)
  
  if (nrow(ego_simple_df) == 0) {
    message("No simplified GO terms for: ", ti$task)
    next
  }
  
  ego_simple_df <- ego_simple_df %>%
    dplyr::arrange(p.adjust, dplyr::desc(Count)) %>%
    dplyr::slice_head(n = SHOW_CATEGORY)
  
  # 把筛后的结果塞回 enrichResult
  ego_final <- ego_simple
  ego_final@result <- ego_simple_df
  
  # foldChange：用 SYMBOL 命名的 attribution 分数
  gene_fc <- df$mean_attr
  names(gene_fc) <- df$gene
  
  # 导出表
  res_export[[ti$task]] <- ego_simple_df %>%
    dplyr::mutate(task = ti$task, subtype = ti$subtype, disease = ti$disease)
  
  # 兼容版 cnetplot
  p <- safe_cnetplot(
    ego_obj = ego_final,
    show_n = min(SHOW_CATEGORY, nrow(ego_final@result)),
    gene_fc = gene_fc
  )
  
  if (is.null(p)) {
    message("cnetplot failed for: ", ti$task)
    next
  }
  
  # 尽量加一个统一色阶；如果对象不支持就跳过
  p <- p +
    ggtitle(paste0(ti$panel, "  ", ti$subtype)) +
    labs(subtitle = ti$disease) +
    theme_cnet(11) +
    theme(
      plot.subtitle = element_text(colour = ti$subtitle_color, face = "bold")
    )
  
  p <- tryCatch(
    p + scale_colour_gradient(low = "#DCEAF4", high = "#2F5D8A"),
    error = function(e) p
  )
  
  plot_list[[ti$task]] <- p
}

if (length(plot_list) == 0) {
  stop("No network plots could be generated.")
}

# =========================
# 6. 拼图
# =========================
plots_final <- plot_list[task_info$task]
plots_final <- plots_final[!sapply(plots_final, is.null)]

fig5C <- wrap_plots(plots_final, ncol = 2) +
  plot_annotation(
    title = "Subtype-specific GO-gene networks for FluA-associated attribution programs",
    theme = theme(
      plot.title = element_text(size = 13.2, face = "bold", hjust = 0, colour = "#111111")
    )
  )

# =========================
# 7. 导出
# =========================
ggsave(
  file.path(out_dir, "Figure5C_cnetplot_main_fixed.png"),
  fig5C, width = 13.2, height = 6.8, dpi = 600, bg = "white"
)

ggsave(
  file.path(out_dir, "Figure5C_cnetplot_main_fixed.pdf"),
  fig5C, width = 13.2, height = 6.8, bg = "white"
)

if (length(res_export) > 0) {
  res_all <- dplyr::bind_rows(res_export)
  write.csv(res_all, file.path(out_dir, "Figure5C_cnetplot_main_fixed_source_data.csv"), row.names = FALSE)
}

cat("Done!\n")
cat("Output dir:", normalizePath(out_dir), "\n")