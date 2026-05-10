library(tidyverse)
library(clusterProfiler)
library(org.Hs.eg.db)
library(scales)

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
TOP_N_GENES <- 120          # 再收一点，提速
TOP_N_TERMS_PER_TASK <- 6
FINAL_SHOW_TERMS <- 12
PADJ_CUTOFF <- 0.20

task_info <- tibble::tibble(
  task = c("G5c_naive_fluA", "unk_epi_fluA", "hillock_covid", "ABC_covid"),
  subtype = c("G5c_naive", "unk_epi", "hillock", "ABC"),
  disease = c("FluA", "FluA", "COVID-19", "COVID-19"),
  display = c("G5c_naive\n(FluA)", "unk_epi\n(FluA)", "hillock\n(COVID-19)", "ABC\n(COVID-19)")
)

REMOVE_DISPLAY_ARTIFACTS <- TRUE
bad_pattern <- "^(RPL|RPS|MRPL|MRPS|MT-|MTRNR|MT-RNR|HBB|HBA)"

# =========================
# 2. universe
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
# 3. ratio parser
# =========================
parse_ratio <- function(x) {
  sapply(strsplit(x, "/"), function(z) as.numeric(z[1]) / as.numeric(z[2]))
}

# =========================
# 4. per-task GO
# =========================
all_res <- list()

for (i in seq_len(nrow(task_info))) {
  ti <- task_info[i, ]
  message("\n===== Running: ", ti$task, " =====")
  
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
  
  message("Mapping genes for ", ti$task, " ...")
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
  
  res <- as.data.frame(ego) %>%
    dplyr::filter(!is.na(p.adjust)) %>%
    dplyr::filter(p.adjust <= PADJ_CUTOFF)
  
  if (nrow(res) == 0) {
    message("No significant GO terms for: ", ti$task)
    next
  }
  
  # 不再跑 simplify，直接取前几个，换速度
  res <- res %>%
    dplyr::mutate(
      GeneRatio_num = parse_ratio(GeneRatio),
      neglog10_padj = -log10(p.adjust),
      task = ti$task,
      subtype = ti$subtype,
      disease = ti$disease,
      display = ti$display
    ) %>%
    dplyr::arrange(p.adjust, dplyr::desc(Count)) %>%
    dplyr::slice_head(n = TOP_N_TERMS_PER_TASK)
  
  all_res[[ti$task]] <- res
  message("Kept ", nrow(res), " terms for ", ti$task)
}

if (length(all_res) == 0) {
  stop("No significant GO BP terms found for any subtype.")
}

res_all <- dplyr::bind_rows(all_res)

# =========================
# 5. matrix data
# =========================
top_terms <- res_all %>%
  dplyr::group_by(Description) %>%
  dplyr::summarise(
    best_p = min(p.adjust, na.rm = TRUE),
    best_ratio = max(GeneRatio_num, na.rm = TRUE),
    best_score = max(neglog10_padj, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(best_p, dplyr::desc(best_ratio)) %>%
  dplyr::slice_head(n = FINAL_SHOW_TERMS) %>%
  dplyr::pull(Description)

plot_df <- res_all %>%
  dplyr::filter(Description %in% top_terms) %>%
  dplyr::mutate(display = factor(display, levels = task_info$display))

plot_df_full <- expand.grid(
  Description = unique(top_terms),
  display = task_info$display,
  stringsAsFactors = FALSE
) %>%
  dplyr::left_join(
    plot_df %>%
      dplyr::select(Description, display, GeneRatio_num, neglog10_padj, Count),
    by = c("Description", "display")
  ) %>%
  dplyr::mutate(display = factor(display, levels = task_info$display))

term_order <- plot_df %>%
  dplyr::group_by(Description) %>%
  dplyr::summarise(best_score = max(neglog10_padj, na.rm = TRUE), .groups = "drop") %>%
  dplyr::arrange(best_score) %>%
  dplyr::pull(Description)

plot_df_full$Description <- factor(plot_df_full$Description, levels = term_order)

wrap_term <- function(x, width = 38) {
  vapply(x, function(s) paste(strwrap(s, width = width), collapse = "\n"), character(1))
}
term_labels <- setNames(wrap_term(term_order, width = 38), term_order)

# =========================
# 6. theme
# =========================
theme_matrix <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      
      panel.grid.major.x = element_line(colour = "#ECEFF2", linewidth = 0.35),
      panel.grid.major.y = element_line(colour = "#F3F5F7", linewidth = 0.30),
      panel.grid.minor   = element_blank(),
      
      axis.title = element_blank(),
      axis.text.x = element_text(size = base_size - 0.1, colour = "#222222", face = "bold"),
      axis.text.y = element_text(size = base_size - 0.2, colour = "#111111"),
      
      legend.title = element_text(size = base_size, face = "bold"),
      legend.text  = element_text(size = base_size - 0.2),
      
      plot.title = element_text(size = base_size + 1.5, face = "bold", hjust = 0, colour = "#111111"),
      plot.subtitle = element_text(size = base_size - 0.1, hjust = 0, colour = "#475569"),
      
      plot.margin = margin(8, 12, 8, 8)
    )
}

# =========================
# 7. plot
# =========================
p <- ggplot(plot_df_full, aes(x = display, y = Description)) +
  geom_point(
    aes(size = GeneRatio_num, fill = neglog10_padj),
    shape = 21,
    colour = "white",
    stroke = 0.35,
    alpha = 0.98,
    na.rm = TRUE
  ) +
  scale_size_continuous(
    range = c(2.5, 10),
    breaks = pretty_breaks(n = 4)
  ) +
  scale_fill_gradient(
    low = "#DCEAF4",
    high = "#2F5D8A",
    na.value = "white"
  ) +
  scale_y_discrete(labels = term_labels) +
  labs(
    title = "Cross-subtype enrichment landscape of attribution-prioritized genes",
    subtitle = "Bubble size indicates GeneRatio; color indicates -log10(FDR). Blank cells mean no significant enrichment."
  ) +
  theme_matrix(11)

# =========================
# 8. export
# =========================
ggsave(
  file.path(out_dir, "Figure5C_enrichment_matrix_fast.png"),
  p, width = 11.8, height = 8.8, dpi = 600, bg = "white"
)

ggsave(
  file.path(out_dir, "Figure5C_enrichment_matrix_fast.pdf"),
  p, width = 11.8, height = 8.8, bg = "white"
)

write.csv(plot_df_full, file.path(out_dir, "Figure5C_enrichment_matrix_fast_source_data.csv"), row.names = FALSE)
write.csv(res_all, file.path(out_dir, "Figure5C_enrichment_matrix_fast_all_terms.csv"), row.names = FALSE)

cat("Done!\n")
cat("Output dir:", normalizePath(out_dir), "\n")