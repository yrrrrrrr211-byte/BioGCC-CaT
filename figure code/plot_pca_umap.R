library(ggplot2)
library(dplyr)
library(patchwork)
library(uwot)
library(grid)

# -----------------------------
# 0. Read data
# -----------------------------
df <- read.csv("E:/Desktop/interpretation_exports/test_xt_flat_merged.csv", stringsAsFactors = FALSE)

class_order <- c("Healthy", "COVID-19", "Influenza A")
df$true_label <- factor(df$true_label, levels = class_order)

feature_cols <- grep("^x_t_", colnames(df), value = TRUE)
X <- as.matrix(df[, feature_cols])
X_scaled <- scale(X)

# -----------------------------
# 1. PCA
# -----------------------------
pca_res <- prcomp(X_scaled, center = FALSE, scale. = FALSE)

pca_df <- data.frame(
  Dim1 = pca_res$x[, 1],
  Dim2 = pca_res$x[, 2],
  true_label = df$true_label
)

pca_var <- summary(pca_res)$importance[2, 1:2] * 100

# -----------------------------
# 2. UMAP
# -----------------------------
set.seed(42)
umap_res <- uwot::umap(
  X_scaled,
  n_neighbors = 30,
  min_dist = 0.30,
  metric = "cosine",
  verbose = TRUE
)

umap_df <- data.frame(
  Dim1 = umap_res[, 1],
  Dim2 = umap_res[, 2],
  true_label = df$true_label
)

# -----------------------------
# 3. Colors
# -----------------------------
my_cols <- c(
  "Healthy" = "#5B8CC0",
  "COVID-19" = "#D08C60",
  "Influenza A" = "#5AA38A"
)

# -----------------------------
# 4. Common theme
# -----------------------------
theme_pub <- theme_classic(base_size = 13, base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    axis.title = element_text(face = "bold", size = 13.5, color = "black"),
    axis.text = element_text(face = "bold", size = 11, color = "black"),
    axis.line = element_line(color = "black", linewidth = 0.7),
    axis.ticks = element_line(color = "black", linewidth = 0.5),
    
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 11, color = "black"),
    legend.box = "horizontal",
    legend.margin = margin(t = -4, b = 0)，
    
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(6, 6, 6, 6)
  )

# -----------------------------
# 5. Helper function
# -----------------------------
make_embed_plot <- function(dat, title_text, xlab_text, ylab_text) {
  ggplot(dat, aes(x = Dim1, y = Dim2)) +
    geom_point(
      aes(color = true_label),
      size = 1.8,
      alpha = 0.80,
      stroke = 0
    ) +
    scale_color_manual(
      values = my_cols,
      guide = guide_legend(
        override.aes = list(size = 3.4, alpha = 1)
      )
    ) +
    labs(
      title = title_text,
      x = xlab_text,
      y = ylab_text
    ) +
    theme_pub
}

# -----------------------------
# 6. Build PCA and UMAP panels
# -----------------------------
p_pca <- make_embed_plot(
  dat = pca_df,
  title_text = "PCA",
  xlab_text = paste0("PC1 (", sprintf("%.1f", pca_var[1]), "%)"),
  ylab_text = paste0("PC2 (", sprintf("%.1f", pca_var[2]), "%)")
)

p_umap <- make_embed_plot(
  dat = umap_df,
  title_text = "UMAP",
  xlab_text = "UMAP1",
  ylab_text = "UMAP2"
)

# -----------------------------
# 7. Assemble
# -----------------------------
Figure3C <- p_pca + p_umap +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

print(Figure3C)

# -----------------------------
# 8. Export
# -----------------------------
ggsave(
  "Figure3C_xt_representation_PCA_UMAP_CLEAN.pdf",
  Figure3C,
  width = 11.4, height = 5.1,
  units = "in",
  device = cairo_pdf,
  bg = "white"
)

ggsave(
  "Figure3C_xt_representation_PCA_UMAP_CLEAN.tiff",
  Figure3C,
  width = 11.4, height = 5.1,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)