

library(DMwR2)
library(dplyr)
library(tidyverse)

# Load your data
df <-  read.csv("wdbc.csv", stringsAsFactors = TRUE)

# Manually specify the feature names
feature_names <- c("radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
                   "smoothness_mean", "compactness_mean", "concavity_mean", 
                   "concave points_mean", "symmetry_mean", "fractal_dimension_mean")

# Ensure correct feature name for "concave points_mean"
feature_names[feature_names == "concave_points_mean"] <- "concave points_mean"

# Extract features and target from the dataframe
features <- df[, feature_names]
target <- df$diagnosis

# Check if features and target are correct
if("diagnosis" %in% names(df)) {
  # Apply SMOTE to balance and increase the sample size
  set.seed(123)  # for reproducibility
  df_smote <- SMOTE(diagnosis ~ ., df, perc.over = 100, k = 5)  # adjust perc.over and k as needed
  
  # Calculate how many more records are needed
  required_records <- 1000 - nrow(df_smote)
  
  # Function to add noise and create synthetic data
  generate_noise_data <- function(original_data, num_records) {
    additional_data <- replicate(num_records, {
      row_to_copy <- original_data[sample(1:nrow(original_data), 1), ]
      noise <- rnorm(ncol(original_data) - 1, mean = 0, sd = apply(original_data[, -ncol(original_data)], 2, sd) * 0.05)
      synthetic_row <- row_to_copy[1:(ncol(original_data) - 1)] + noise
      c(synthetic_row, row_to_copy[ncol(original_data)])
    })
    return(t(additional_data))
  }
  
  # Create additional synthetic data
  additional_df <- as.data.frame(generate_noise_data(df_smote, required_records))
  names(additional_df) <- names(df_smote)
  
  # Combine with the SMOTE dataframe
  final_df <- rbind(df_smote, additional_df)
  
  # Shuffle the dataset
  set.seed(123)
  final_df <- final_df[sample(1:nrow(final_df)), ]
  
  # Reset row names to avoid any future indexing issues
  rownames(final_df) <- NULL
  
  # Optionally, save the augmented dataset
  write.csv(final_df, "new_dataset.csv", row.names = FALSE)
  
  # Print a message
  print("New dataset with 1000 records created and saved.")
} else {
  print("Error: 'diagnosis' not found in dataframe columns.")
}