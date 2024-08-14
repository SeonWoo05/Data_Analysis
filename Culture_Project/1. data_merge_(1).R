library(dplyr)

getwd()
setwd('C:/Users/My-Notebook/SODA/Culture_Project/code')
directory_path <- "C:/Users/My-Notebook/SODA/Culture_Project/code/Normal_data"

file_list <- list.files(directory_path, pattern = "*.csv", full.names = TRUE)
data_list <- lapply(file_list, read.csv, stringsAsFactors = FALSE)
combined_data <- do.call(rbind, data_list)

write.csv(combined_data, "combined_data.csv", row.names = FALSE, fileEncoding = "EUC-KR")