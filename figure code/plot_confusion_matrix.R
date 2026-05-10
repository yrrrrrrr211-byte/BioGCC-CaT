# =========================================================
# Figure 2. Composite confusion matrices
# A. M0
# B. M5
# C. 5-fold aggregated
# FINAL STABLE VERSION
# =========================================================

library(ggplot2)
library(dplyr)
library(scales)
library(patchwork)
library(grid)

# -----------------------------
# 0. Global settings
# -----------------------------
class_names <- c("Healthy", "COVID-19", "Influenza A")
base_family_use <- "sans"   # 想换字体可改成 "Arial"

# -----------------------------
# 1. Input matrices
# -----------------------------
# M0
cm_M0 <- matrix(
  c(
    515, 24, 19,
    22, 463, 40,
    52, 91, 1698
  ),
  nrow = 3,
  byrow = TRUE
)

# M5
cm_M5 <- matrix(
  c(
    503, 20, 35,
    12, 464, 49,
    10, 36, 1795
  ),
  nrow = 3,
  byrow = TRUE
)

# 5-fold CV
cm_fold1 <- matrix(
  c(
    1073, 22, 28,
    53, 1003, 73,
    102, 75, 3430
  ),
  nrow = 3,
  byrow = TRUE
)

cm_fold2 <- matrix(
  c(
    1063, 26, 35,
    64, 958, 106,
    80, 42, 3485
  ),
  nrow = 3,
  byrow = TRUE
)

cm_fold3 <- matrix(
  c(
    1046, 37, 40,
    31, 1006, 91,
    76, 71, 3461
  ),
  nrow = 3,
  byrow = TRUE
)

cm_fold4 <- matrix(
  c(
    1005, 19, 49,
    60, 968, 100,
    63, 45, 3499
  ),
  nrow = 3,
  byrow = TRUE
)

cm_fold5 <- matrix(
  c(
    1026, 61, 36,
    16, 1034, 78,
    63, 150, 3394
  ),
  nrow = 3,
  byrow = TRUE
)

# Aggregated 5-fold confusion matrix
cm_agg <- cm_fold1 + cm_fold2 + cm_fold3 + cm_fold4 + cm_fold5

cat("Aggregated 5-fold confusion matrix:\n")
print(cm_agg)
# 应为：
#      [,1] [,2]  [,3]
# [1,] 5213  165   188
# [2,]  224 4969   448
# [3,]  384  383 17269

# -----------------------------
# 2. Helper: matrix -> tidy df
# -----------------------------
mat_to_df <- function(mat) {
  df <- expand.grid(True = 1:3, Pred = 1:3)
  
  df$Count <- mapply(function(i, j) mat[i, j], df$True, df$Pred)
  
  row_totals <- rowSums(mat)
  df$Total <- row_totals[df$True]
  df$Proportion <- df$Count / df$Total
  df$Percent <- df$Proportion * 100
  df$Label <- sprintf("%.1f%%\n(%d/%d)", df$Percent, df$Count, df$Total)
  df$TextColor <- ifelse(df$Proportion >= 0.50, "white", "black")
  
  # x 轴：左到右 Healthy, COVID-19, Influenza A
  df$x <- df$Pred
  
  # y 轴：上到下 Healthy, COVID-19, Influenza A
  df$y <- 4 - df$True
  
  df
}

# -----------------------------
# 3. Helper: single confusion matrix panel (NO title here)
# -----------------------------
plot_one_confmat <- function(mat,
                             show_y_title = FALSE,
                             show_x_title = FALSE,
                             show_legend = FALSE) {
  
  df <- mat_to_df(mat)
  
  p <- ggplot(df, aes(x = x, y = y, fill = Proportion)) +
    geom_tile(width = 0.98, height = 0.98, color = "#F0F0F0", linewidth = 0.9) +
    geom_text(
      aes(label = Label, color = TextColor),
      size = 4.0,
      fontface = "bold",
      lineheight = 0.92,
      family = base_family_use
    ) +
    scale_color_identity() +
    scale_fill_gradientn(
      colours = c("#F7FBFF", "#DEEBF7", "#C6DBEF", "#6BAED6", "#2171B5"),
      limits = c(0, 1),
      breaks = c(0, 0.25, 0.50, 0.75, 1.00),
      labels = percent_format(accuracy = 1),
      name = "Row-normalized\nproportion"
    ) +
    scale_x_continuous(
      breaks = 1:3,
      labels = class_names,
      limits = c(0.5, 3.5),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      breaks = c(3, 2, 1),
      labels = class_names,
      limits = c(0.5, 3.5),
      expand = c(0, 0)
    ) +
    labs(
      x = if (show_x_title) "Predicted label" else NULL,
      y = if (show_y_title) "True label" else NULL
    ) +
    theme_bw(base_size = 13, base_family = base_family_use) +
    theme(
      aspect.ratio = 1,
      panel.grid = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      
      axis.text.x = element_text(
        angle = 15, hjust = 1, vjust = 1,
        face = "bold", color = "black", size = 11.0,
        margin = margin(t = 2)
      ),
      axis.text.y = element_text(
        face = "bold", color = "black", size = 11.5
      ),
      axis.title.x = element_text(
        face = "bold", color = "black", size = 14,
        margin = margin(t = 8)
      ),
      axis.title.y = element_text(
        face = "bold", color = "black", size = 14,
        margin = margin(r = 8)
      ),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      
      legend.title = element_text(
        face = "bold", color = "black", size = 10.5
      ),
      legend.text = element_text(
        color = "black", size = 10
      ),
      legend.key.height = unit(0.72, "cm"),
      legend.key.width  = unit(0.40, "cm"),
      
      plot.margin = margin(0, 4, 4, 4)
    )
  
  if (!show_legend) {
    p <- p + theme(legend.position = "none")
  }
  
  p
}

# -----------------------------
# 4. Helper: title strip (separate small row above each panel)
# -----------------------------
make_title_strip <- function(label_text) {
  ggplot() +
    annotate(
      "text",
      x = 0.5, y = 0.5,
      label = label_text,
      fontface = "bold",
      size = 6.0,
      family = base_family_use
    ) +
    xlim(0, 1) + ylim(0, 1) +
    theme_void() +
    theme(
      plot.margin = margin(0, 0, -10, 0)
    )
}

# -----------------------------
# 5. Build main panels
# -----------------------------
pA_main <- plot_one_confmat(
  mat = cm_M0,
  show_y_title = TRUE,
  show_x_title = FALSE,
  show_legend = FALSE
)

pB_main <- plot_one_confmat(
  mat = cm_M5,
  show_y_title = FALSE,
  show_x_title = TRUE,
  show_legend = FALSE
)

pC_main <- plot_one_confmat(
  mat = cm_agg,
  show_y_title = FALSE,
  show_x_title = FALSE,
  show_legend = TRUE
)

# -----------------------------
# 6. Add title strips
# -----------------------------
colA <- (make_title_strip("A. M0") / pA_main) +
  plot_layout(heights = c(0.07, 1))

colB <- (make_title_strip("B. M5") / pB_main) +
  plot_layout(heights = c(0.07, 1))

colC <- (make_title_strip("C. 5-fold aggregated") / pC_main) +
  plot_layout(heights = c(0.07, 1))

# -----------------------------
# 7. Assemble final figure
# -----------------------------
Figure2 <- (colA | colB | colC) +
  plot_layout(widths = c(1, 1, 1.10))

print(Figure2)

# -----------------------------
# 8. Export
# -----------------------------
ggsave(
  filename = "Figure2_confusion_matrix_composite_FINAL.pdf",
  plot = Figure2,
  width = 13.8, height = 4.9,
  units = "in",
  device = cairo_pdf,
  bg = "white"
)

ggsave(
  filename = "Figure2_confusion_matrix_composite_FINAL.tiff",
  plot = Figure2,
  width = 13.8, height = 4.9,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)