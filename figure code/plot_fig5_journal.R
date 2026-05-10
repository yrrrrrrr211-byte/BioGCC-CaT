library(tidyverse)
library(patchwork)
# =========================
# 0. 输入
# =========================
in_file <- "E:/Desktop/cell level/fig5A_top_genes_for_plot.csv"
out_dir <- "Figure5A_journal"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# =========================
# 1. 读数据
# =========================
df <- read.csv(in_file, stringsAsFactors = FALSE, check.names = FALSE)

# =========================
# 2. 展示层清洗（只影响作图，不改 raw 结果）
# =========================
REMOVE_DISPLAY_ARTIFACTS <- TRUE

if (REMOVE_DISPLAY_ARTIFACTS) {
  bad_pattern <- "^(RPL|RPS|MRPL|MRPS|MT-|MTRNR|MT-RNR|HBB|HBA)"
  df <- df %>%
    filter(!grepl(bad_pattern, gene, ignore.case = TRUE))
}

# 每个 panel 重新取 top 10
df <- df %>%
  group_by(task) %>%
  arrange(desc(mean_attr), .by_group = TRUE) %>%
  slice_head(n = 10) %>%
  ungroup()

# =========================
# 3. 面板顺序与标题
# =========================
panel_order <- c(
  "G5c_naive_fluA",
  "unk_epi_fluA",
  "hillock_covid",
  "ABC_covid"
)

panel_title_map <- c(
  "G5c_naive_fluA" = "A  G5c_naive | FluA evidence",
  "unk_epi_fluA"   = "B  unk_epi | FluA evidence",
  "hillock_covid"  = "C  hillock | COVID-19 evidence",
  "ABC_covid"      = "D  ABC | COVID-19 evidence"
)

panel_color_map <- c(
  "G5c_naive_fluA" = "#2CA02C",
  "unk_epi_fluA"   = "#2CA02C",
  "hillock_covid"  = "#D62728",
  "ABC_covid"      = "#D62728"
)

# 统一 x 轴范围
xmax <- max(df$mean_attr, na.rm = TRUE) * 1.18

# =========================
# 4. 主题
# =========================
theme_pub <- function(base_size = 11) {
  theme_bw(base_size = base_size) +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.border     = element_rect(colour = "black", linewidth = 0.35),
      axis.line.x      = element_line(colour = "black", linewidth = 0.25),
      axis.ticks       = element_line(colour = "black", linewidth = 0.25),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(colour = "grey90", linewidth = 0.25),
      axis.title       = element_text(size = base_size + 0.3, colour = "black"),
      axis.text.x      = element_text(size = base_size, colour = "black"),
      axis.text.y      = element_text(size = base_size, colour = "black", face = "italic"),
      plot.title       = element_text(size = base_size + 1.8, face = "bold", hjust = 0),
      plot.margin      = margin(6, 10, 6, 6)
    )
}

# =========================
# 5. 单 panel 绘图函数
# =========================
make_panel <- function(task_name, show_xlab = TRUE) {
  d <- df %>%
    filter(task == task_name) %>%
    arrange(mean_attr) %>%
    mutate(gene = factor(gene, levels = gene))
  
  p_col <- panel_color_map[[task_name]]
  p_title <- panel_title_map[[task_name]]
  
  ggplot(d, aes(x = mean_attr, y = gene)) +
    geom_segment(
      aes(x = 0, xend = mean_attr, yend = gene),
      colour = "grey80", linewidth = 0.9
    ) +
    geom_point(
      shape = 21, size = 4.2, stroke = 0.35,
      fill = p_col, colour = "black"
    ) +
    geom_text(
      aes(label = sprintf("%.3f", mean_attr)),
      hjust = -0.15, size = 3.1, colour = "grey20"
    ) +
    coord_cartesian(xlim = c(0, xmax), clip = "off") +
    labs(
      x = ifelse(show_xlab, "Mean attribution (Gradient × Input)", NULL),
      y = NULL
    ) +
    ggtitle(p_title) +
    theme_pub(11) +
    theme(
      plot.title = element_text(colour = p_col, face = "bold"),
      axis.text.y = element_text(face = "italic"),
      axis.text.x = element_text(face = "bold")
    )
}

# =========================
# 6. 画四个 panel
# =========================
p1 <- make_panel("G5c_naive_fluA", show_xlab = FALSE)
p2 <- make_panel("unk_epi_fluA",   show_xlab = FALSE)
p3 <- make_panel("hillock_covid",  show_xlab = TRUE)
p4 <- make_panel("ABC_covid",      show_xlab = TRUE)

fig5A <- (p1 | p2) / (p3 | p4) +
  plot_layout(guides = "collect") +
  plot_annotation(
    theme = theme(
      plot.margin = margin(8, 8, 8, 8)
    )
  )

# =========================
# 7. 导出
# =========================
ggsave(
  file.path(out_dir, "Figure5A_journal.png"),
  fig5A, width = 13.5, height = 9.8, dpi = 500, bg = "white"
)

ggsave(
  file.path(out_dir, "Figure5A_journal.pdf"),
  fig5A, width = 13.5, height = 9.8, bg = "white"
)

write.csv(df, file.path(out_dir, "Figure5A_source_data.csv"), row.names = FALSE)

cat("Done!\n")
cat("Output dir:", normalizePath(out_dir), "\n")