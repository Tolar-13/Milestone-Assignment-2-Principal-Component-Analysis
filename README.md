# PCA and Logistic Regression Analysis on the Breast Cancer Dataset

## Project Overview

The objective of this project is to address the increasing number of referrals at the Anderson Cancer Center by identifying essential variables for securing donor funding. We utilize the Breast Cancer dataset from `sklearn.datasets` and implement Principal Component Analysis (PCA) to achieve this goal. Additionally, logistic regression is used as a step to predict cancer diagnosis based on the reduced dataset.

## Key Objectives

1. **Principal Component Analysis (PCA)**: Use PCA to identify the most significant variables in the dataset that can aid in decision-making and donor funding efforts.
   
2. **Dimensionality Reduction**: Reduce the dataset to 2 principal components to simplify analysis and visualization, while retaining as much essential information as possible.

3. **Visualization of PCA Components**: Plot the reduced components to gain insights into the data distribution and differentiate between malignant and benign cancer diagnoses.

4. **Logistic Regression**: Implement logistic regression to predict cancer diagnosis using the reduced PCA components, providing an additional layer of analysis.

## Dataset Description

The Breast Cancer dataset contains data about breast cancer cell nuclei derived from digitized images of fine needle aspirate (FNA) biopsies. The dataset consists of 569 instances with 30 numeric features describing various characteristics of the cell nuclei, such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension. The target variable indicates whether the cancer is malignant (coded as 1) or benign (coded as 0).

### Features

The dataset includes 30 continuous variables that are categorized into three groups:

- **Mean Measurements**: Mean values for features like radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

- **Error Measurements**: Standard error for the same features mentioned above.

- **Worst Measurements**: Worst (or largest) values for the above features computed across all cell nuclei.

### Target Variable

- **Target**: Binary classification where 1 represents malignant cancer and 0 represents benign cancer.

## Data Preprocessing

Before applying PCA, several preprocessing steps are undertaken to ensure data quality and consistency:

1. **Handling Missing Values**: We verify that the dataset is complete and contains no missing values.
   
2. **Removing Duplicates**: Duplicate entries, if any, are removed to avoid biases in analysis.

3. **Normalization**: Features are standardized to ensure that PCA treats all features equally, as they may have different units and scales. This step is crucial since PCA is sensitive to the variance of each variable.

## PCA Implementation

### Principal Component Analysis (PCA)

PCA is applied to the normalized dataset to identify and reduce the number of dimensions while retaining the most important features that explain the maximum variance in the dataset. This process involves:

- **Explained Variance Ratio**: Determining the amount of variance explained by each principal component. This helps in understanding which components capture the most information.

- **Cumulative Explained Variance**: Calculating the cumulative explained variance to decide how many components should be retained to achieve a satisfactory level of variance explanation.

### Essential Variables Identified

Through PCA, we identified key variables that contribute significantly to the variance in the data and can potentially influence donor funding strategies. These variables have high loadings on the first two principal components, indicating their importance in distinguishing between malignant and benign cases.

| **Variable Name**             | **PC1 Loading** | **PC2 Loading** | **Contribution**                  |
|-------------------------------|-----------------|-----------------|----------------------------------|
| **mean_radius**               | High Positive   | Moderate Negative | Strong influence on tumor size   |
| **mean_texture**              | Moderate Positive | Moderate Positive | Indicates consistency of cells   |
| **mean_perimeter**            | High Positive   | Moderate Negative | Related to tumor boundary size   |
| **mean_area**                 | High Positive   | Moderate Negative | Reflects overall tumor size      |
| **mean_smoothness**           | Moderate Positive | Moderate Positive | Relates to cell regularity       |
| **mean_compactness**          | High Positive   | Moderate Positive | Shape feature indicating density |
| **mean_concavity**            | High Positive   | High Positive    | Curvature of tumor boundary      |
| **mean_concave points**       | High Positive   | High Positive    | Points on the perimeter          |
| **worst_radius**              | High Positive   | Moderate Negative | Largest tumor dimension          |
| **worst_perimeter**           | High Positive   | Moderate Negative | Largest tumor boundary size      |
| **worst_area**                | High Positive   | Moderate Negative | Largest tumor area               |
| **worst_concave points**      | High Positive   | High Positive    | Most concave points on perimeter |
| **worst_texture**             | Moderate Positive | Moderate Positive | Largest deviation in texture     |

#### Interpretation

- **Size-Related Features**: Variables like `mean_radius`, `mean_area`, `worst_radius`, etc., are crucial as they indicate tumor size and boundary, which are significant indicators of malignancy. Larger tumors are often associated with malignant cases, making these features vital for donor funding strategies targeting early detection and treatment.

- **Shape Complexity Features**: Variables such as `mean_concavity` and `worst_concave points` describe the tumor's shape complexity. Higher values in these features typically indicate malignancy due to irregular tumor boundaries, making them essential for research focused on precise tumor characterization.

- **Textural Features**: Features like `mean_texture` and `mean_smoothness` provide insights into the consistency and regularity of tumor cells. Their contribution to PC1 highlights their role in identifying differences between benign and malignant cases, crucial for funding proposals aimed at improving diagnostic accuracy.

## Dimensionality Reduction

The dataset is reduced to two principal components to facilitate visualization and analysis. This reduction allows us to simplify the complex 30-dimensional space into a 2-dimensional representation without losing critical information. The first two principal components capture approximately 62.8% of the total variance, making them highly informative for analysis purposes.

## Visualizing PCA Components

Visualization of the two principal components provides a clear picture of how data points are distributed across the malignant and benign classes. A scatter plot is created, with the two principal components on the axes, and data points colored based on their respective class labels.

### Visualization Analysis

The scatter plot reveals distinct clusters representing malignant and benign tumors. The clear separation between these clusters indicates that the two principal components effectively differentiate between the two classes. This visualization helps in identifying key variables that contribute to this separation and provides insights into potential factors influencing donor funding decisions.

#### Key Insights

- **Data Separation**: The plot shows a clear separation along PC1, with malignant cases predominantly located on the positive side. This indicates that variables with high PC1 loadings are crucial for classification.

- **Impact on Donor Funding**: The identified essential variables should be emphasized in donor funding strategies, focusing on areas like early detection, advanced imaging, and precision treatment.

- **Visual Representation**:
  
  - **Clusters**: Benign tumors (blue), positioned primarily in the lower-left quadrant, indicate lower values in both PC1 and PC2, which represent smaller and less irregular tumors while malignant tumors (red), spread out along the positive side of PC1, represent larger and more complex tumors. This illustrates the effectiveness of PCA in identifying essential variables.
    
  - **Annotations**: Arrows and text indicate the contribution of key variables to PC1 and PC2, emphasizing their importance in donor funding strategies. PC1 (Horizontal Spread) with high variance due to size-related features, indicates the need for funding towards early diagnosis technologies while PC2 (Vertical Spread) with additional variance capturing shape and texture nuances, highlights the potential for research into precise imaging solutions.

## Logistic Regression

logistic regression is implemented using the reduced dataset (2 PCA components) to predict cancer diagnoses. The steps involved in this analysis include:

1. **Data Splitting**: Dividing the dataset into training and testing sets to evaluate the model's performance.

2. **Model Training**: Training a logistic regression model on the 2-component PCA data to learn the relationship between the components and the target variable.

3. **Prediction and Evaluation**: The model predicts the cancer diagnosis for the test set, and its performance is evaluated using metrics like accuracy, confusion matrix, and classification report. This evaluation provides insights into the model's predictive capabilities and potential areas for improvement.

## Conclusions

### Key Findings

1. **PCA Efficiency**: PCA successfully reduces the dimensionality of the dataset while retaining essential information. The first two principal components explain a significant portion of the variance, making them valuable for analysis and decision-making.

2. **Visualization Insights**: The visualization of PCA components reveals clear separation between malignant and benign cases, highlighting the effectiveness of PCA in identifying essential variables.

3. **Logistic Regression Performance**: The logistic regression model shows reasonable accuracy of 99% in predicting cancer diagnoses using just the two principal components. This indicates that the reduced dataset still contains sufficient information for classification tasks.

### Recommendations

- **Donor Funding Strategy**: We should focus on the variables with high loadings in the first two PCA components, as they contribute most to data variability and class separation. These variables can be emphasized in our donor funding strategies.

## Files in Repository

- **README.md**: Detailed project documentation and instructions.

- **pca_analysis.ipynb**: Jupyter Notebook containing the complete analysis and visualization of the PCA and logistic regression implementation.

- **pca_analysis.py**: Python script with the code implementation for PCA and logistic regression.

- **pca_components.png**: Visualization of the 2 PCA components highlighting the separation between malignant and benign cases.

## Running the Code

### Option 1: Jupyter Notebook

1. Open `pca_analysis.ipynb` in Jupyter Notebook.
2. Execute the cells sequentially to run the analysis and view the results.

### Option 2: Python Script

1. Ensure you have Python and the required libraries installed (`numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`).
2. Run the script using the command: `python pca_analysis.py`.

## Dependencies

- Python 3.x
- numpy
- pandas
- matplotlib
- seaborn
- sklearn

## Conclusion

This project provides a comprehensive analysis of essential variables from the Breast Cancer dataset using PCA. The identification of these variables helps Anderson Cancer Center tailor donor funding strategies, focusing on early detection, precise diagnosis, and effective treatment. Logistic regression further validates the effectiveness of the PCA components in classifying cancer diagnoses, offering a robust approach to decision-making and funding efforts.

By understanding the key factors that contribute to cancer diagnosis, we at Anderson Cancer Center can effectively communicate the importance of these variables to potential donors, supporting initiatives that advance research and improve patient outcomes.
