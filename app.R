library(shiny)
library(shinydashboard)
library(e1071)
library(randomForest)
library(class)
library(DT)
library(corrplot)

source("model_training.R")

# Load the models
logisticModel <- readRDS("logisticModel.rds")
svmModel <- readRDS("svmModel.rds")
randomForestModel <- readRDS("randomForestModel.rds")
knnModel <- readRDS("knnModel.rds")
modelAccuracies <- readRDS("modelAccuracies.rds")
modelMetrics <- readRDS("modelMetrics.rds")

extractMetrics <- function(cm) {
  data.frame(
    Accuracy = cm$overall['Accuracy'],
    Sensitivity = cm$byClass['Sensitivity'],
    Specificity = cm$byClass['Specificity']
  )
}

logisticMetrics <- extractMetrics(modelMetrics$logistic)
svmMetrics <- extractMetrics(modelMetrics$svm)
randomForestMetrics <- extractMetrics(modelMetrics$randomForest)
knnMetrics <- extractMetrics(modelMetrics$knn)

combinedMetrics <- rbind(
  cbind(Model = "Logistic Regression", logisticMetrics),
  cbind(Model = "SVM", svmMetrics),
  cbind(Model = "Random Forest", randomForestMetrics),
  cbind(Model = "KNN", knnMetrics)
)

ui <- dashboardPage(
  dashboardHeader(
    title = "Breast Cancer Prediction",
    titleWidth = 650,
    tags$li(class = "dropdown",
            tags$a(href = "https://www.youtube.com/playlist?list=PL6wLL_RojB5xNOhe2OTSd-DPkMLVY9DfB",
                   icon("youtube"), " My Channel", target = "_blank")),
    tags$li(class = "dropdown",
            tags$a(href = "https://www.linkedin.com/in/abhinav-agrawal-pmp%C2%AE-safe%C2%AE-5-agilist-csm%C2%AE-5720309",
                   icon("linkedin"), " My Profile", target = "_blank")),
    tags$li(class = "dropdown",
            tags$a(href = "https://github.com/aagarw30/R-Shiny-Dashboards/tree/main/USArrestDashboard",
                   icon("github"), " Source Code", target = "_blank"))
  ),
  dashboardSidebar(
    sidebarMenu(
      menuItem("About", tabName = "about", icon = icon("info-circle")),
      menuItem("Data", tabName = "data", icon = icon("database")),
      menuItem("Visualization", tabName = "visualization", icon = icon("chart-bar")),  # Visualization menu item
      menuItem("Model Prediction", tabName = "model_prediction", icon = icon("project-diagram"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(
        tabName = "data",
        tabBox(
          id = "dataTabBox",
          width = 12,
          tabPanel("Data", dataTableOutput("dataT"), icon = icon("table")),
          tabPanel("Structure", dataTableOutput("structureTable"), icon = icon("uncharted")),
          tabPanel("Summary Stats", dataTableOutput("summary"), icon = icon("chart-pie"))
        )
      ),
      tabItem(
        tabName = "visualization",
        tabBox(
          width = 12,
          tabPanel("Feature Distributions", plotOutput("featureDistributionsPlot")),
          tabPanel("Correlation Heatmap", plotOutput("correlationHeatmapPlot")),
          tabPanel("Feature Importance",plotOutput("featureImportancePlot"))
        )
      ),
      tabItem(
        tabName = "model_prediction",
        tabBox(
          id = "modelPredictionTabs",
          width = 12,
          tabPanel("Prediction",
                   fluidRow(
                     box(title = "Input Features", status = "primary", solidHeader = TRUE, collapsible = TRUE,
                         width = 12,
                         numericInput("radius_mean", "Radius Mean", value = 14),
                         numericInput("texture_mean", "Texture Mean", value = 20),
                         numericInput("perimeter_mean", "Perimeter Mean", value = 85),
                         numericInput("area_mean", "Area Mean", value = 550),
                         numericInput("smoothness_mean", "Smoothness Mean", value = 0.1),
                         numericInput("compactness_mean", "Compactness Mean", value = 0.1),
                         numericInput("concavity_mean", "Concavity Mean", value = 0.1),
                         numericInput("concave_points_mean", "Concave Points Mean", value = 0.05),
                         numericInput("symmetry_mean", "Symmetry Mean", value = 0.18),
                         numericInput("fractal_dimension_mean", "Fractal Dimension Mean", value = 0.06),
                         actionButton("predict", "Predict")
                     )
                   )),
          tabPanel("Results",
                   fluidRow(
                     valueBoxOutput("logisticBox"),
                     valueBoxOutput("svmBox"),
                     valueBoxOutput("randomForestBox"),
                     valueBoxOutput("knnBox")
                   )),
          tabPanel("Model Performance",
                   fluidRow(
                     box(title = "Model Metrics", status = "primary", solidHeader = TRUE, collapsible = TRUE,
                         DTOutput("metricsTable")
                     )
                   ))
        )
      )
    )
  )
)

server <- function(input, output, session) {
  observeEvent(input$predict, {
    inputData <- data.frame(
      radius_mean = as.numeric(input$radius_mean),
      texture_mean = as.numeric(input$texture_mean),
      perimeter_mean = as.numeric(input$perimeter_mean),
      area_mean = as.numeric(input$area_mean),
      smoothness_mean = as.numeric(input$smoothness_mean),
      compactness_mean = as.numeric(input$compactness_mean),
      concavity_mean = as.numeric(input$concavity_mean),
      concave_points_mean = as.numeric(input$concave_points_mean),
      symmetry_mean = as.numeric(input$symmetry_mean),
      fractal_dimension_mean = as.numeric(input$fractal_dimension_mean)
    )
    
    # Make predictions
    logisticPred <- predict(logisticModel, newdata = inputData, type = "response")
    logisticPred <- ifelse(logisticPred > 0.5, "Malignant", "Benign")
    
    svmPred <- predict(svmModel, newdata = inputData, type = "response")
    svmPred <- ifelse(svmPred == 1, "Malignant", "Benign")
    
    randomForestPred <- predict(randomForestModel, newdata = inputData, type = "response")
    randomForestPred <- ifelse(randomForestPred == 1, "Malignant", "Benign")
    
    knnPred <- predict(knnModel, newdata = inputData, type = "class")
    knnPred <- ifelse(knnPred == 1, "Malignant", "Benign")
    
    updateTabsetPanel(session = session, inputId = "modelPredictionTabs", selected = "Results")
    
    # Update UI
    output$logisticBox <- renderValueBox({
      valueBox(
        "Logistic Regression",
        HTML(paste("<div style='font-size: 18px;'>Prediction: <b>", logisticPred, "</b>","<br>Accuracy: <b>", sprintf("%.2f%%", modelAccuracies["logistic"] * 100), "</b> </div>")),
        icon = icon("dna"),
        color = "green"
      )
    })
    output$svmBox <- renderValueBox({
      valueBox(
        "SVM",
        HTML(paste("<div style='font-size: 18px;'>Prediction: <b>", svmPred, "</b>","<br>Accuracy: <b>", sprintf("%.2f%%", modelAccuracies["svm"] * 100), "</b> </div>")),
        icon = icon("microscope"),
        color = "blue"
      )
    })
    output$randomForestBox <- renderValueBox({
      valueBox(
        "Random Forest",
        HTML(paste("<div style='font-size: 18px;'>Prediction: <b>", logisticPred, "</b>","<br>Accuracy: <b>", sprintf("%.2f%%", modelAccuracies["logistic"] * 100), "</b> </div>")),
        icon = icon("leaf"),
        color = "red"
      )
    })
    output$knnBox <- renderValueBox({
      valueBox(
        "KNN",
        HTML(paste("Prediction: <b>", knnPred, "</b>","<br>Accuracy: <b>", sprintf("%.2f%%", modelAccuracies["knn"] * 100), "</b> </div>")),
        icon = icon("users"),
        color = "yellow"
      )
    })
  })
  output$metricsTable <- renderDT({
    datatable(combinedMetrics, options = list(pageLength = 10), rownames = FALSE)
  })
  
  output$dataT <- renderDataTable({
    datatable(data_final, 
              options = list(
                pageLength = 10, 
                autoWidth = FALSE,
                scrollX = TRUE,    # Enable horizontal scrolling
                scrollY = "650px",  # Set vertical scrolling area height
                scrollCollapse = TRUE,# Optional: Allow scroll area to shrink if the content is smaller
                paging = FALSE    # Disable pagination to show all data on a single page
              ), class="display"
    )
  })
  
  output$structureTable <- renderDataTable({
    datatable(df_structure, 
              options = list(
                pageLength = 10, 
                autoWidth = FALSE,
                scrollX = TRUE,    # Enable horizontal scrolling
                scrollY = "650px",  # Set vertical scrolling area height
                scrollCollapse = TRUE,# Optional: Allow scroll area to shrink if the content is smaller
                paging = FALSE
                ), class="display")
  })
  
  output$summary <- renderDataTable({
    selectedVars <- c("radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
                      "smoothness_mean", "compactness_mean","concavity_mean", "concave_points_mean","symmetry_mean","fractal_dimension_mean")
    stats <- calculateSummaryStats(data_final, selectedVars)
    print(stats)
    datatable(stats, 
              options = list(
                pageLength = 10, autoWidth = FALSE,scrollX=TRUE,scrollY = "650px",scrollCollapse = TRUE,paging = FALSE), class = "display"
              )
  })
  
  output$featureDistributionsPlot <- renderPlot({
    # Prepare the data for plotting by gathering the selected features into long format
    selectedVars <- c("radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
                      "smoothness_mean", "compactness_mean","concavity_mean", "concave_points_mean","symmetry_mean","fractal_dimension_mean")
    longData <- reshape2::melt(data_final[, selectedVars], variable.name = "Feature", value.name = "Value")
    
    # Create the plot with a consistent and attractive theme
    p <- ggplot(longData, aes(x = Value, fill = Feature)) +
      geom_histogram(bins = 20, color = "black", alpha = 0.7) +
      scale_fill_brewer(palette = "Blues", guide = FALSE) +  # Using a color brewer palette for aesthetic fill
      facet_wrap(~ Feature, scales = "free") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1), 
            strip.background = element_rect(fill = "lightblue"),
            strip.text = element_text(size = 10),
            legend.position = "none") +  # Hide the legend if not needed
      labs(title = "Feature Distributions", x = "", y = "Count")
    
    print(p)
  })
  
  
  output$correlationHeatmapPlot <- renderPlot({
    # Ensure you're only including numeric columns for the correlation calculation
    selected_features <- data_final[, c("radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
                                        "smoothness_mean", "compactness_mean","concavity_mean", "concave_points_mean","symmetry_mean","fractal_dimension_mean","diagnosis")]
    numeric_features <- selected_features[, sapply(selected_features, is.numeric)]
    
    # Calculate correlation on numeric features
    M <- cor(numeric_features, use = "complete.obs")
    corrplot(M, method = "color", type = "upper", order = "hclust",
             addCoef.col = "black", tl.col = "black", tl.srt = 45,
             title = "Feature Correlation Heatmap")
  })
  
  # Assuming randomForestModel is your Random Forest model object loaded from an RDS file

  # Ensure correct extraction of feature names and importance values
  if(ncol(randomForestModel$importance) > 0) {
    feature_importance <- data.frame(
      Feature = rownames(randomForestModel$importance),
      Importance = randomForestModel$importance[, "MeanDecreaseGini"]
    )
  } else {
    stop("No importance data available in the model.")
  }
  
  
  output$featureImportancePlot <- renderPlot({
    ggplot(feature_importance, aes(x = reorder(Feature, Importance), y = Importance, fill = Importance)) +
      geom_bar(stat = "identity") +  # Use fill from the aesthetic mapping
      scale_fill_gradient(low = "lightblue", high = "darkblue") +  # Gradient fill from light to dark blue
      coord_flip() +  # Horizontal layout for better readability of feature names
      labs(
        title = "Feature Importance",
        x = "Features",
        y = "Importance (Mean Decrease Gini)"
      ) +
      theme_minimal() +
      theme(
        axis.title.x = element_text(face = "bold", size = 12),  # Make x-axis label bold
        axis.title.y = element_text(face = "bold", size = 12),  # Make y-axis label bold
        plot.title = element_text(face = "bold", size = 14)     # Optionally, make the plot title bold
      )
  })
  
}

shinyApp(ui, server)