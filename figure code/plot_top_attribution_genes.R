library(tidyverse)
library(patchwork)

# =========================
# 1. 路径设置
# =========================
run_dir <- "E:/Desktop"
attr_dir <- file.path(run_dir, "holdout_attribution")
out_dir  <- file.path(run_dir, "paper_figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

covid_file <- file.path(attr_dir, "heldout_covid_gene_attr.csv")
flua_file  <- file.path(attr_dir, "heldout_flua_gene_attr.csv")

TOP_N <- 15

# =========================
# 2. 读取数据
# =========================
covid_df <- read.csv(covid_file, stringsAsFactors = FALSE)
flua_df  <- read.csv(flua_file, stringsAsFactors = FALSE)

covid_top <- covid_df %>%
  arrange(desc(attr_mean_signed)) %>%
  slice_head(n = TOP_N) %>%
  mutate(channel = "COVID-19")

flua_top <- flua_df %>%
  arrange(desc(attr_mean_signed)) %>%
  slice_head(n = TOP_N) %>%
  mutate(channel = "Influenza A")

top_combined <- bind_rows(covid_top, flua_top)

write.csv(
  top_combined,
  file.path(out_dir, "Fig4A_top_attribution_genes_source.csv"),
  row.names = FALSE
)

# =========================
# 3. 主题设置
# =========================
theme_paper <- function() {
  theme_bw(base_size = 12) +
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "grey88", linewidth = 0.4),
      panel.border = element_rect(color = "black", linewidth = 0.6),
      axis.text = element_text(color = "black"),
      axis.title = element_text(color = "black"),
      plot.title = element_text(face = "bold", hjust = 0.5, size = 12.5),
      legend.position = "none",
      plot.margin = margin(8, 10, 8, 10)
    )
}

# =========================
# 4. 配色
# =========================
covid_fill <- "#2C7FB8"   # 深蓝
flua_fill  <- "#7FCDBB"   # 青蓝
label_col  <- "grey20"

# =========================
# 5. 作图函数
# =========================
plot_attr_bar <- function(df, panel_title, fill_color) {
  df_plot <- df %>%
    mutate(gene = factor(gene, levels = rev(gene)))
  
  ggplot(df_plot, aes(x = attr_mean_signed, y = gene)) +
    geom_col(fill = fill_color, width = 0.72) +
    geom_text(
      aes(label = sprintf("%.3f", attr_mean_signed)),
      hjust = -0.08,
      size = 3.1,
      color = label_col
    ) +
    scale_x_continuous(expand = expansion(mult = c(0.00, 0.16))) +
    labs(
      title = panel_title,
      x = "Mean signed attribution score",
      y = NULL
    ) +
    theme_paper()
}

p1 <- plot_attr_bar(
  covid_top,
  "A. COVID-19 evidence-associated genes",
  covid_fill
)

p2 <- plot_attr_bar(
  flua_top,
  "B. Influenza A evidence-associated genes",
  flua_fill
)

fig4a <- p1 + p2 + plot_layout(ncol = 2)

# =========================
# 6. 导出图片
# =========================
png_file <- file.path(out_dir, "Fig4A_top_attribution_genes_blue.png")
pdf_file <- file.path(out_dir, "Fig4A_top_attribution_genes_blue.pdf")
svg_file <- file.path(out_dir, "Fig4A_top_attribution_genes_blue.svg")

ggsave(png_file, fig4a, width = 11.5, height = 6.2, dpi = 400, bg = "white")
ggsave(pdf_file, fig4a, width = 11.5, height = 6.2, bg = "white")
ggsave(svg_file, fig4a, width = 11.5, height = 6.2, bg = "white")

# =========================
# 7. 导出图注
# =========================
caption <- paste(
  "Figure 4A. Top attribution genes associated with the COVID-19 and Influenza A evidence channels.",
  "Genes were ranked according to the mean signed attribution score derived from the final held-out M5 model.",
  "Higher positive scores indicate stronger positive contributions to the corresponding class-specific evidence channel."
)

writeLines(caption, file.path(out_dir, "Fig4A_caption.txt"))

cat("Done!\n")
cat("Saved files:\n")
cat(" -", png_file, "\n")
cat(" -", pdf_file, "\n")
cat(" -", svg_file, "\n")
cat(" -", file.path(out_dir, "Fig4A_top_attribution_genes_source.csv"), "\n")
cat(" -", file.path(out_dir, "Fig4A_caption.txt"), "\n")