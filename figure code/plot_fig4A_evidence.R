library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(scales)
library(grid)

# -----------------------------
# 0. Read data
# -----------------------------
df <- read.csv("E:/Desktop/interpretation_exports/test_evidence_merged.csv", stringsAsFactors = FALSE)
class_order <- c("Healthy", "COVID-19", "Influenza A")
df$true_label <- factor(df$true_label, levels = class_order)

# -----------------------------
# 1. Build matched / competitor
# -----------------------------
df2 <- df %>%
  mutate(
    matched_evidence = case_when(
      true_label == "Healthy" ~ healthy_evidence,
      true_label == "COVID-19" ~ covid_evidence,
      true_label == "Influenza A" ~ flua_evidence
    ),
    competitor_evidence = case_when(
      true_label == "Healthy" ~ pmax(covid_evidence, flua_evidence),
      true_label == "COVID-19" ~ pmax(healthy_evidence, flua_evidence),
      true_label == "Influenza A" ~ pmax(healthy_evidence, covid_evidence)
    ),
    evidence_margin = matched_evidence - competitor_evidence,
    matched_is_max = matched_evidence > competitor_evidence
  )

plot_df <- df2 %>%
  select(true_label, matched_evidence, competitor_evidence) %>%
  pivot_longer(
    cols = c(matched_evidence, competitor_evidence),
    names_to = "EvidenceType",
    values_to = "Evidence"
  ) %>%
  mutate(
    EvidenceType = factor(
      EvidenceType,
      levels = c("matched_evidence", "competitor_evidence"),
      labels = c("Matched", "Strongest competitor")
    )
  )

# -----------------------------
# 2. Stats
# -----------------------------
fmt_p <- function(p) {
  if (p < 1e-4) return("P < 1e-4")
  paste0("P = ", formatC(p, format = "f", digits = 4))
}

stats_df <- df2 %>%
  group_by(true_label) %>%
  summarise(
    p_value = wilcox.test(matched_evidence, competitor_evidence, paired = TRUE)$p.value,
    matched_rate = mean(matched_is_max) * 100,
    median_margin = median(evidence_margin),
    .groups = "drop"
  ) %>%
  mutate(
    p_label = sapply(p_value, fmt_p),
    stat_line = paste0(
      p_label,
      "   |   Max-match rate: ", sprintf("%.1f%%", matched_rate),
      "   |   Median Δ = ", sprintf("%.2f", median_margin)
    )
  )

# -----------------------------
# 3. Top strip
# -----------------------------
make_stat_strip <- function(title_text, stat_text) {
  ggplot() +
    annotate(
      "text",
      x = 0.5, y = 0.68,
      label = title_text,
      fontface = "bold",
      size = 6.1,
      family = "sans"
    ) +
    annotate(
      "text",
      x = 0.5, y = 0.16,
      label = stat_text,
      size = 3.35,
      color = "#555555",
      family = "sans"
    ) +
    xlim(0, 1) + ylim(0, 1) +
    theme_void() +
    theme(plot.margin = margin(0, 0, -3, 0))
}

# -----------------------------
# 4. Main panel
# -----------------------------
make_main_panel <- function(one_class, show_y_title = FALSE, show_x_title = FALSE) {
  sub_df <- plot_df %>% filter(true_label == one_class)
  
  ggplot(sub_df, aes(x = EvidenceType, y = Evidence, fill = EvidenceType)) +
    geom_violin(
      width = 0.86,
      scale = "width",
      trim = FALSE,
      alpha = 0.92,
      color = NA
    ) +
    geom_boxplot(
      width = 0.16,
      outlier.shape = NA,
      linewidth = 0.55,
      fill = "white",
      color = "black"
    ) +
    stat_summary(
      fun = median,
      geom = "point",
      shape = 23,
      size = 2.2,
      fill = "white",
      color = "black",
      stroke = 0.45
    ) +
    scale_fill_manual(
      values = c("Matched" = "#3F7FBC", "Strongest competitor" = "#D4D9E0")
    ) +
    coord_cartesian(ylim = c(0, 9.3), clip = "off") +
    labs(
      x = if (show_x_title) "Evidence type" else NULL,
      y = if (show_y_title) "Evidence score" else NULL
    ) +
    theme_bw(base_size = 13, base_family = "sans") +
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      
      axis.text.x = element_text(face = "bold", size = 11, color = "black"),
      axis.text.y = element_text(face = "bold", size = 11, color = "black"),
      axis.title.x = element_text(face = "bold", size = 13.5, color = "black", margin = margin(t = 8)),
      axis.title.y = element_text(face = "bold", size = 13.5, color = "black", margin = margin(r = 8)),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      
      legend.position = "none",
      plot.margin = margin(0, 4, 4, 4)
    )
}

# -----------------------------
# 5. Assemble
# -----------------------------
col_healthy <- (
  make_stat_strip("Healthy", stats_df$stat_line[stats_df$true_label == "Healthy"]) /
    make_main_panel("Healthy", show_y_title = TRUE, show_x_title = FALSE)
) + plot_layout(heights = c(0.16, 1))

col_covid <- (
  make_stat_strip("COVID-19", stats_df$stat_line[stats_df$true_label == "COVID-19"]) /
    make_main_panel("COVID-19", show_y_title = FALSE, show_x_title = TRUE)
) + plot_layout(heights = c(0.16, 1))

col_flua <- (
  make_stat_strip("Influenza A", stats_df$stat_line[stats_df$true_label == "Influenza A"]) /
    make_main_panel("Influenza A", show_y_title = FALSE, show_x_title = FALSE)
) + plot_layout(heights = c(0.16, 1))

Figure3A <- col_healthy | col_covid | col_flua

print(Figure3A)

ggsave(
  "Figure3A_matched_vs_competing_evidence_polished.pdf",
  Figure3A,
  width = 13.2, height = 5.0,
  units = "in",
  device = cairo_pdf,
  bg = "white"
)

ggsave(
  "Figure3A_matched_vs_competing_evidence_polished.tiff",
  Figure3A,
  width = 13.2, height = 5.0,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)

write.csv(stats_df, "Figure3A_evidence_stats.csv", row.names = FALSE)
print(stats_df)