library(tidyverse)
library(patchwork)
library(scales)

# =========================
# 0. 输入
# =========================
in_file <- "E:/Desktop/cell level/fig5A_top_genes_for_plot.csv"

# 输出目录 = 输入文件所在目录
out_dir <- dirname(in_file)

# =========================
# 1. 读数据
# =========================
df <- read.csv(in_file, stringsAsFactors = FALSE, check.names = FALSE)

# =========================
# 2. 展示层清洗（仅用于主文图）
# =========================
# 不改 raw 结果，只清理最容易把图带脏的基因
REMOVE_DISPLAY_ARTIFACTS <- TRUE

if (REMOVE_DISPLAY_ARTIFACTS) {
  bad_pattern <- "^(RPL|RPS|MRPL|MRPS|MT-|MTRNR|MT-RNR|HBB|HBA)"
  df <- df %>%
    filter(!grepl(bad_pattern, gene, ignore.case = TRUE))
}

# 每个 panel 重新取 top 8
df <- df %>%
  group_by(task) %>%
  arrange(desc(mean_attr), .by_group = TRUE) %>%
  slice_head(n = 8) %>%
  ungroup()

# =========================
# 3. panel 信息
# =========================
panel_order <- c(
  "G5c_naive_fluA",
  "unk_epi_fluA",
  "hillock_covid",
  "ABC_covid"
)

panel_info <- tibble(
  task = panel_order,
  letter = c("A", "B", "C", "D"),
  subtype = c("G5c_naive", "unk_epi", "hillock", "ABC"),
  disease = c("FluA", "FluA", "COVID-19", "COVID-19"),
  color = c("#148A8A", "#148A8A", "#3F6B99", "#3F6B99")
)

df <- df %>%
  filter(task %in% panel_order) %>%
  left_join(panel_info, by = "task") %>%
  mutate(task = factor(task, levels = panel_order))

# 统一 x 轴
xmax <- max(df$mean_attr, na.rm = TRUE) * 1.12
xbreaks <- pretty(c(0, xmax), n = 4)

# =========================
# 4. 极简正文主题
# =========================
theme_rankdot <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      
      panel.grid.major.y = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_line(colour = "#E8EDF2", linewidth = 0.35),
      
      axis.title.y = element_blank(),
      axis.title.x = element_text(size = base_size + 0.2, colour = "#222222"),
      axis.text.y  = element_text(size = base_size, colour = "#111111", face = "italic"),
      axis.text.x  = element_text(size = base_size - 0.2, colour = "#222222"),
      
      axis.line.x  = element_line(colour = "#222222", linewidth = 0.28),
      axis.ticks.x = element_line(colour = "#222222", linewidth = 0.28),
      
      plot.title    = element_text(size = base_size + 1.3, face = "bold", hjust = 0, colour = "#111111"),
      plot.subtitle = element_text(size = base_size - 0.1, face = "bold", hjust = 0),
      
      plot.margin = margin(8, 10, 8, 8)
    )
}

# =========================
# 5. 单 panel 函数
# =========================
make_panel <- function(task_name, show_x = TRUE) {
  d <- df %>%
    filter(task == task_name) %>%
    arrange(mean_attr) %>%
    mutate(gene = factor(gene, levels = gene))
  
  info <- panel_info %>% filter(task == task_name)
  p_col <- info$color[[1]]
  p_title <- paste0(info$letter[[1]], "  ", info$subtype[[1]])
  p_sub <- info$disease[[1]]
  
  x_lab <- if (show_x) "Mean attribution (Gradient × Input)" else NULL
  
  ggplot(d, aes(x = mean_attr, y = gene)) +
    # 很淡的基线
    geom_segment(
      aes(x = 0, xend = mean_attr, yend = gene),
      colour = "#C9D1D9",
      linewidth = 0.9,
      lineend = "round"
    ) +
    # 外圈点
    geom_point(
      size = 4.4,
      shape = 21,
      fill = "white",
      colour = p_col,
      stroke = 1.2
    ) +
    # 内芯点
    geom_point(
      size = 2.0,
      shape = 16,
      colour = p_col
    ) +
    scale_x_continuous(
      limits = c(0, xmax),
      breaks = xbreaks,
      labels = number_format(accuracy = 0.001),
      expand = c(0, 0)
    ) +
    labs(
      x = x_lab,
      y = NULL,
      title = p_title,
      subtitle = p_sub
    ) +
    theme_rankdot(11) +
    theme(
      plot.subtitle = element_text(colour = p_col),
      axis.title.x = if (show_x) element_text() else element_blank(),
      axis.text.x  = if (show_x) element_text(face = "bold") else element_blank(),
      axis.ticks.x = if (show_x) element_line() else element_blank()
    )
}

# =========================
# 6. 拼图
# =========================
p1 <- make_panel("G5c_naive_fluA", show_x = FALSE)
p2 <- make_panel("unk_epi_fluA",   show_x = FALSE)
p3 <- make_panel("hillock_covid",  show_x = TRUE)
p4 <- make_panel("ABC_covid",      show_x = TRUE)

fig5A <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    title = "Subtype-specific attribution genes underlying disease-biased evidence patterns",
    theme = theme(
      plot.title = element_text(size = 13.2, face = "bold", hjust = 0, colour = "#111111")
    )
  )

# =========================
# 7. 导出（直接到输入文件目录）
# =========================
ggsave(
  file.path(out_dir, "Figure5A_rankdot.png"),
  fig5A, width = 12.6, height = 8.6, dpi = 600, bg = "white"
)

ggsave(
  file.path(out_dir, "Figure5A_rankdot.pdf"),
  fig5A, width = 12.6, height = 8.6, bg = "white"
)

write.csv(df, file.path(out_dir, "Figure5A_rankdot_source_data.csv"), row.names = FALSE)

cat("Done!\n")
cat("Output dir:", normalizePath(out_dir), "\n")