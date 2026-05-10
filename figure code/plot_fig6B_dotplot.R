library(tidyverse)
library(patchwork)
library(scales)

# =========================
# 0. 路径
# =========================
base_dir  <- "E:/Desktop/cell level"

meta_file <- file.path(base_dir, "test_main_with_celltype.csv")
expr_file <- file.path(base_dir, "testing_sample.csv")
gene_file <- file.path(base_dir, "rna_name.csv")
gene_set_file <- file.path(base_dir, "fig5A_top_genes_for_plot.csv")

# 输出目录 = 输入目录
out_dir <- base_dir

# =========================
# 1. 参数
# =========================
TOP_N_PER_TASK <- 5

task_info <- tibble(
  task = c("G5c_naive_fluA", "unk_epi_fluA", "hillock_covid", "ABC_covid"),
  panel = c("A", "B", "C", "D"),
  subtype = c("G5c_naive", "unk_epi", "hillock", "ABC"),
  disease = c("FluA", "FluA", "COVID-19", "COVID-19"),
  status = c("fluA", "fluA", "COVID-19", "COVID-19"),
  y_true = c(2, 2, 1, 1),
  y_pred = c(2, 2, 1, 1)
)

subtype_display_order <- c("G5c_naive", "unk_epi", "hillock", "ABC")
subtype_display_labels <- c(
  "G5c_naive\n(FluA)",
  "unk_epi\n(FluA)",
  "hillock\n(COVID-19)",
  "ABC\n(COVID-19)"
)

# =========================
# 2. 读数据
# =========================
meta <- read.csv(meta_file, stringsAsFactors = FALSE, check.names = FALSE)

# 表达矩阵可能有点大，优先用 data.table::fread，没有就 read.csv
if (requireNamespace("data.table", quietly = TRUE)) {
  expr <- data.table::fread(expr_file, header = FALSE, data.table = FALSE)
  expr <- as.matrix(expr)
} else {
  expr <- as.matrix(read.csv(expr_file, header = FALSE, check.names = FALSE))
}

gene_names <- read.csv(gene_file, header = FALSE, stringsAsFactors = FALSE)[, 1]

stopifnot(nrow(meta) == nrow(expr))
stopifnot(length(gene_names) == ncol(expr))

colnames(expr) <- gene_names

# log1p 变换，做表达图更稳
expr_log <- log1p(expr)

# =========================
# 3. 读 Figure 5A 基因表，并取每个 task 的 top genes
# =========================
gene_df <- read.csv(gene_set_file, stringsAsFactors = FALSE, check.names = FALSE)

# 展示层再次清洗
bad_pattern <- "^(RPL|RPS|MRPL|MRPS|MT-|MTRNR|MT-RNR|HBB|HBA)"
gene_df <- gene_df %>%
  filter(!grepl(bad_pattern, gene, ignore.case = TRUE))

gene_df <- gene_df %>%
  filter(task %in% task_info$task) %>%
  group_by(task) %>%
  arrange(desc(mean_attr), .by_group = TRUE) %>%
  slice_head(n = TOP_N_PER_TASK) %>%
  ungroup()

# =========================
# 4. 定义 4 个 subtype population（与 attribution 一致）
# =========================
cell_groups <- list()

for (i in seq_len(nrow(task_info))) {
  ti <- task_info[i, ]
  
  idx <- which(
    meta$cellType == ti$subtype &
      meta$status   == ti$status &
      meta$y_true   == ti$y_true &
      meta$y_pred   == ti$y_pred
  )
  
  cell_groups[[ti$subtype]] <- idx
  message(ti$subtype, ": ", length(idx), " cells")
}

# =========================
# 5. 计算 dot plot 数据
# =========================
plot_list <- list()

for (task_name in task_info$task) {
  genes_this <- gene_df %>%
    filter(task == task_name) %>%
    pull(gene)
  
  genes_this <- unique(genes_this)
  
  for (g in genes_this) {
    if (!g %in% colnames(expr_log)) next
    
    vals <- c()
    for (subtype_name in subtype_display_order) {
      idx <- cell_groups[[subtype_name]]
      
      if (length(idx) == 0) {
        avg_expr <- NA
        pct_expr <- NA
      } else {
        v_raw <- expr[idx, g]
        v_log <- expr_log[idx, g]
        
        avg_expr <- mean(v_log, na.rm = TRUE)
        pct_expr <- mean(v_raw > 0, na.rm = TRUE) * 100
      }
      
      vals <- bind_rows(
        vals,
        tibble(
          task = task_name,
          subtype_ref = subtype_name,
          gene = g,
          avg_expr = avg_expr,
          pct_expr = pct_expr
        )
      )
    }
    
    plot_list[[length(plot_list) + 1]] <- vals
  }
}

plot_df <- bind_rows(plot_list)

# =========================
# 6. 对每个 gene 在 4 个 subtype 间做 z-score
# =========================
plot_df <- plot_df %>%
  group_by(task, gene) %>%
  mutate(
    expr_z = ifelse(
      sd(avg_expr, na.rm = TRUE) > 0,
      as.numeric(scale(avg_expr)),
      0
    )
  ) %>%
  ungroup()

# 补 panel 信息
plot_df <- plot_df %>%
  left_join(task_info %>% select(task, panel, subtype, disease), by = "task") %>%
  mutate(
    subtype_ref = factor(subtype_ref, levels = subtype_display_order, labels = subtype_display_labels)
  )

# 为每个 panel 的 gene 排序
plot_df <- plot_df %>%
  group_by(task, gene) %>%
  summarise(mean_attr_proxy = mean(avg_expr, na.rm = TRUE), .groups = "drop") %>%
  group_by(task) %>%
  arrange(mean_attr_proxy, .by_group = TRUE) %>%
  mutate(gene_order = row_number()) %>%
  ungroup() %>%
  select(task, gene, gene_order) %>%
  right_join(plot_df, by = c("task", "gene"))

# =========================
# 7. 主题
# =========================
theme_dotpanel <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      
      panel.grid.major.x = element_line(colour = "#ECEFF2", linewidth = 0.35),
      panel.grid.major.y = element_blank(),
      panel.grid.minor   = element_blank(),
      
      axis.title.y = element_blank(),
      axis.title.x = element_blank(),
      axis.text.x  = element_text(size = base_size - 0.1, colour = "#222222"),
      axis.text.y  = element_text(size = base_size, colour = "#111111", face = "italic"),
      
      plot.title    = element_text(size = base_size + 1.2, face = "bold", hjust = 0, colour = "#111111"),
      plot.subtitle = element_text(size = base_size - 0.1, face = "bold", hjust = 0),
      
      legend.title = element_text(size = base_size, face = "bold"),
      legend.text  = element_text(size = base_size - 0.2),
      
      plot.margin = margin(8, 10, 8, 8)
    )
}

# =========================
# 8. 单 panel 函数
# =========================
make_panel <- function(task_name, show_legend = FALSE) {
  d <- plot_df %>%
    filter(task == task_name) %>%
    mutate(
      gene = reorder(gene, gene_order)
    )
  
  info <- task_info %>% filter(task == task_name)
  
  p_col <- ifelse(info$disease[[1]] == "FluA", "#148A8A", "#3F6B99")
  p_title <- paste0(info$panel[[1]], "  ", info$subtype[[1]])
  p_sub <- info$disease[[1]]
  
  p <- ggplot(d, aes(x = subtype_ref, y = gene)) +
    geom_point(
      aes(size = pct_expr, fill = expr_z),
      shape = 21,
      colour = "white",
      stroke = 0.35,
      alpha = 0.98
    ) +
    scale_size_continuous(
      range = c(2.2, 10),
      breaks = c(25, 50, 75, 100),
      limits = c(0, 100)
    ) +
    scale_fill_gradient2(
      low = "#F4F7FA",
      mid = "#A9C4D6",
      high = "#2F5D8A",
      midpoint = 0
    ) +
    labs(
      title = p_title,
      subtitle = p_sub,
      x = NULL,
      y = NULL,
      size = "% cells\nexpressing",
      fill = "Scaled\nexpression"
    ) +
    theme_dotpanel(11) +
    theme(
      plot.subtitle = element_text(colour = p_col),
      axis.text.x = element_text(face = "bold"),
      legend.position = if (show_legend) "right" else "none"
    )
  
  return(p)
}

# =========================
# 9. 拼图
# =========================
p1 <- make_panel("G5c_naive_fluA", show_legend = FALSE)
p2 <- make_panel("unk_epi_fluA",   show_legend = FALSE)
p3 <- make_panel("hillock_covid",  show_legend = FALSE)
p4 <- make_panel("ABC_covid",      show_legend = TRUE)

fig5B <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    title = "Subtype-specific expression patterns of attribution-prioritized genes",
    theme = theme(
      plot.title = element_text(size = 13.2, face = "bold", hjust = 0, colour = "#111111")
    )
  )

# =========================
# 10. 导出
# =========================
ggsave(
  file.path(out_dir, "Figure5B_dotplot.png"),
  fig5B, width = 12.8, height = 9.4, dpi = 600, bg = "white"
)

ggsave(
  file.path(out_dir, "Figure5B_dotplot.pdf"),
  fig5B, width = 12.8, height = 9.4, bg = "white"
)

write.csv(plot_df, file.path(out_dir, "Figure5B_dotplot_source_data.csv"), row.names = FALSE)

cat("Done!\n")
cat("Output dir:", normalizePath(out_dir), "\n")