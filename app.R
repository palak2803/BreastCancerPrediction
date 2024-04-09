#load required packages
library(shiny)
library(shinydashboard)
library(e1071)
library(randomForest)
library(class)
library(DT)
library(corrplot)

#Reference the model_training.R script.
source("model_training.R")

# Load the models and extract accuracies, Sensitivity and Specificity
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

# User interface components
ui <- dashboardPage(
  dashboardHeader(
    title = "Breast Cancer Prediction",
    titleWidth = 650,
    tags$li(class = "dropdown",
            tags$a(href = "https://github.com/palak2803/BreastCancerPrediction",
                   icon("github"), "Source Code", class = "btn-primary")),
    tags$li(class = "dropdown",
            downloadButton("downloadReport", "Download Report", class = "btn-primary"))
  ),
  dashboardSidebar(
    sidebarMenu(
      menuItem("About", tabName = "about", icon = icon("info-circle")),
      menuItem("Data", tabName = "data", icon = icon("database")),
      menuItem("Visualization", tabName = "visualization", icon = icon("chart-bar")), 
      menuItem("Model Prediction", tabName = "model_prediction", icon = icon("project-diagram"))
    )
  ),
  dashboardBody(
    tags$head(
      tags$style(HTML('
                /* Settinh Header background color */
                .main-header .navbar, .main-header .logo {
                    background-color: #c2185b !important; 
                }

                /* Setting Sidebar toggle and other navigation bar elements */
                .main-header .navbar .sidebar-toggle, .main-header .navbar .navbar-custom-menu a {
                    color: #ffffff !important; 
                }

                /* Adjusting Sidebar color */
                .skin-blue .main-sidebar, .skin-blue .left-side {
                    background-color: #c2185b !important; 
                }

                /* Override button colors for consistency */
                .btn-primary {
                    background-color: #c2185b; 
                    border-color: #ad1457;
                }
                .btn-primary:hover {
                    background-color: #ad1457; 
                    border-color: #880e4f;
                }

                /* Active menu item in the sidebar */
                .sidebar-menu > li.active > a {
                    background-color: #c2185b !important;
                    border-left-color: #d81b60 !important;
                }

                /* General link colors */
                a {
                    color: #880e4f; 
                }
                
                /* Tab and box headers in the main content */
                .nav-tabs-custom > .nav-tabs > li.active {
                    border-top-color: #c2185b;
                }
                .box.box-solid > .box-header {
                    background-color: #f06292;
                    border-color: #f06292;
                }
                .nav-tabs-custom > .nav-tabs > li {
                    background: #f8bbd0;
                    border-top-color: #c2185b;
                }
                
                .dataTables_wrapper .dataTables_paginate .paginate_button {
                    background-color: #f8bbd0;
                    color: #880e4f !important;
                }
                .dataTables_wrapper .dataTables_paginate .paginate_button:hover {
                    background-color: #ec407a;
                    color: #ffffff !important;
                }
                .dataTables_wrapper .dataTables_filter input,
                .dataTables_wrapper .dataTables_length select {
                    border-color: #ec407a; /* Pink border for input and select */
                }
                table.dataTable thead th, table.dataTable thead td {
                    background-color: #ec407a !important; /* Pink header for tables */
                    color: #ffffff;
                }
                table.dataTable {
                    border-color: #ec407a; /* Pink border for table */
                }
            
            '))
    ),
    
    tabItems(
      tabItem(
        tabName = "about",
        h3("About the Breast Cancer Prediction App"),
        h5("This application is developed as a part of AMOD5250H final project which assists users in predicting breast cancer severity based on diagnostic measurements of the tumor characteristics extracted from images. 
            It integrates advanced machine learning algorithms such as Logistic Regression, Support vector machine, Random Forest and, KNN and compares their performace metrics against the Breast Cancer Wisconsin Dataset (WBCD). 
            This tool can help in early detection and enhances the decision-making process, significantly improving patient outcomes."),
        fluidRow(
          column(6, img(src = "about_image.jpeg", height = "300px", style = "width:100%;")),
          column(6, img(src = "tumor_image.jpeg", height = "300px", style = "width:100%;"))
        ),
        h3("Understanding Breast Cancer: Benign vs. Malignant"),
        h5("Breast cancer can manifest as benign or malignant tumors. 
           Benign tumors are non-cancerous and do not spread to other parts of the body, often easy to treat and unlikely to recur. Malignant tumors, however, are cancerous and can invade nearby tissues or metastasize to distant areas, requiring more aggressive treatment. 
           Early and accurate distinction between these types is vital for effective treatment."),
        h3("How it works"),
        h5("Users can navigate through various tabs to view data visualizations, enter new diagnostic measurements to receive predictions, and compare model performances. 
            The application dynamically updates the visualizations based on user interactions, providing a tailored analytical experience. 
            Additionally, users can access the application's source code via the 'Source Code' button in the top-right corner and download the application report using the 'Download Report' button."),
        h3("Scientific Basis"),
        h5("The models are trained on well-curated data from the Wisconsin Breast Cancer Database, which includes measurements such as the radius, texture, perimeter, and area of breast mass samples. Each model has been fine-tuned to offer high accuracy and reliability, reflecting the latest advancements in machine learning and statistical analysis.")
      ),
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
                     box(status = "primary", solidHeader = FALSE, collapsible = FALSE,
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
          tabPanel("Model Performance", dataTableOutput("metricsTable"), icon = icon("table")),
        )
      )
    )
  )
)

#Server component that handles the backend processing of the application
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
    
    # Logic to create the prediction results boxes
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
  # Logic to generate the metrics table
  output$metricsTable <- renderDT({
    datatable(combinedMetrics,options = list(
      pageLength = 10, 
      autoWidth = FALSE,
      scrollX = TRUE,   
      scrollY = "650px", 
      scrollCollapse = TRUE,
      paging = FALSE   
    ), class="display" )
  })
  
  #logic to generate the raw data table
  output$dataT <- renderDataTable({
    datatable(data_final, 
              options = list(
                pageLength = 10, 
                autoWidth = FALSE,
                scrollX = TRUE,   
                scrollY = "650px", 
                scrollCollapse = TRUE,
                paging = FALSE  
              ), class="display"
    )
  })
  
  #logic to generate the structure table
  output$structureTable <- renderDataTable({
    datatable(df_structure, 
              options = list(
                pageLength = 10, 
                autoWidth = FALSE,
                scrollX = TRUE,   
                scrollY = "650px",  
                scrollCollapse = TRUE,
                paging = FALSE
                ), class="display")
  })
  
  # Logic to render the summary table
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
  
  #Logic to handle the feature distribution plot
  output$featureDistributionsPlot <- renderPlot({
    # Prepare the data for plotting by gathering the selected features into long format
    selectedVars <- c("radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
                      "smoothness_mean", "compactness_mean","concavity_mean", "concave_points_mean","symmetry_mean","fractal_dimension_mean")
    longData <- reshape2::melt(data_final[, selectedVars], variable.name = "Feature", value.name = "Value")
    
    # Create the plot with a consistent and attractive theme
    p <- ggplot(longData, aes(x = Value, fill = Feature)) +
      geom_histogram(bins = 20, color = "black", alpha = 0.7) +
      scale_fill_manual(values = colorRampPalette(c("#FFC0CB", "#FF69B4", "#C71585"))(200))  +  
      facet_wrap(~ Feature, scales = "free") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1), 
            strip.background = element_rect(fill = "#e0bbd2"),
            strip.text = element_text(size = 10),
            legend.position = "none") +  
      labs(title = "Feature Distributions", x = "", y = "Count")
    
    print(p)
  })
  
  #Logic to generate correlation heatmap
  output$correlationHeatmapPlot <- renderPlot({
  
    selected_features <- data_final[, c("radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
                                        "smoothness_mean", "compactness_mean","concavity_mean", "concave_points_mean","symmetry_mean","fractal_dimension_mean","diagnosis")]
    numeric_features <- selected_features[, sapply(selected_features, is.numeric)]
    
    # Calculate correlation on numeric features
    M <- cor(numeric_features, use = "complete.obs")
    corrplot(M, 
             method = "color", 
             type = "upper", 
             order = "hclust",
             col = colorRampPalette(c("#FFC0CB", "#FF69B4", "#C71585"))(200),
             addCoef.col = "black", 
             tl.col = "black", 
             tl.srt = 45
            )
  })
  
  #logic to generate the feature importance bar plot
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
      scale_fill_gradient(low = "#e0bbd2", high = "#a3386c") +  # Gradient fill from light to dark blue
      coord_flip() +  # Horizontal layout for better readability of feature names
      labs(
        title = "Feature Importance",
        x = "Features",
        y = "Importance (Mean Decrease Gini)"
      ) +
      theme_minimal() +
      theme(
        axis.title.x = element_text(face = "bold", size = 12),  
        axis.title.y = element_text(face = "bold", size = 12), 
        plot.title = element_text(face = "bold", size = 14)    
      )
  })
  
  # logic to implement the download report functionality
  output$downloadReport <- downloadHandler(
    filename = function() {
      "Report.pdf" 
    },
    content = function(file) {
      file.copy("Report.pdf", file) 
    }
  )
  
}

# main application entry-point
shinyApp(ui, server)