install.packages("dplyr")

getwd()
setwd("C:/Users/My-Notebook/SODA/dept_project") # 경로 수정 -> 현재 디렉토리로

# 22년도 데이터
library(dplyr)
directory_path <- "C:/Users/My-Notebook/SODA/dept_project/data_22"
file_list <- list.files(directory_path, pattern = "*.csv", full.names = TRUE)
column_names <- c("strd_yymm", "strd_date", "strd_tizn_val", "swst_id", "swst_nm", "swst_lgd_cdn_val", "swst_ltd_cdn_val", "usr_num")
data_list <- lapply(file_list, function(file) {
  data <- read.csv(file, header = FALSE, stringsAsFactors = FALSE)
  colnames(data) <- c(column_names, "A", "B")
  data <- data[, column_names]
  sorted_data <- data[order(data$strd_date, data$strd_tizn_val), ]
  return(sorted_data)
})
combined_data <- do.call(rbind, data_list)
write.csv(combined_data, "C:/Users/My-Notebook/SODA/dept_project/data/combined_data_22.csv", row.names = FALSE, fileEncoding = "EUC-KR")

# 23년도 데이터
directory_path <- "C:/Users/My-Notebook/SODA/dept_project/data_23"
file_list <- list.files(directory_path, pattern = "*.csv", full.names = TRUE)
column_names <- c("strd_yymm", "strd_date", "strd_tizn_val", "swst_id", "swst_nm", "swst_lgd_cdn_val", "swst_ltd_cdn_val", "usr_num")
data_list <- lapply(file_list, function(file) {
  first_row <- read.csv(file, nrows = 1, header = FALSE, stringsAsFactors = FALSE)
  if (all(first_row %in% column_names)){
    data <- read.csv(file, header = TRUE, stringsAsFactors = FALSE, fileEncoding = "EUC-KR")
  }
  else{
    data <- read.csv(file, header = FALSE, stringsAsFactors = FALSE)
    colnames(data) <- c(column_names, "A", "B")
    data <- data[, column_names]
  }
  sorted_data <- data[order(data$strd_date, data$strd_tizn_val), ]
  return(sorted_data)
})
combined_data <- do.call(rbind, data_list)
write.csv(combined_data, "C:/Users/My-Notebook/SODA/dept_project/data/combined_data_23.csv", row.names = FALSE, fileEncoding = "EUC-KR")

# 24년도 데이터
directory_path <- "C:/Users/My-Notebook/SODA/dept_project/data_24"
file_list <- list.files(directory_path, pattern = "*.csv", full.names = TRUE)
data_list <- lapply(file_list, function(file) {
  data <- read.csv(file, header = TRUE, stringsAsFactors = FALSE, fileEncoding = "EUC-KR")
  sorted_data <- data[order(data$strd_date, data$strd_tizn_val), ]
  return(sorted_data)
})
combined_data <- do.call(rbind, data_list)
write.csv(combined_data, "C:/Users/My-Notebook/SODA/dept_project/data/combined_data_24.csv", row.names = FALSE, fileEncoding = "EUC-KR")

