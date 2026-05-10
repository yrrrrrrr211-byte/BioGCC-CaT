# Example data structure

Raw expression matrices are not included because of file size. To reproduce preprocessing and training, place the downloaded GEO GSE176269 files under the following structure:

```text
raw_data/
+-- expression/
|   `-- CovidStudy_rawCounts_061721.txt
`-- metadata/
    `-- CovidStudy_phenotype_061721.txt
```

After preprocessing, the scripts expect a data directory containing:

```text
data/
+-- training_sample.csv
+-- training_label.csv
+-- validation_sample.csv
+-- validation_label.csv
+-- testing_sample.csv
+-- testing_label.csv
`-- rna_name.csv
```
