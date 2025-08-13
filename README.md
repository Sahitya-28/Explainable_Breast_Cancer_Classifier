# Explainable Breast Cancer Classification
### *Making AI decisions transparent in healthcare*
This project uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset to train a Random Forest classifier and then provides an interactive explainability dashboard for examining both global and local model behaviors.
## Repository Structure
 ``` explainable-breast-cancer-ml/
├── notebooks/
│   └── breast_cancer_analysis.ipynb      # Main analysis notebook with code and explanations.
├── requirements.txt                      # Lists all the necessary Python libraries for this project.
└── README.md                             # Provides a detailed overview of the project.
```
## Project Workflow

### 1. Environment Verification
This project ensures **reproducibility** by displaying the versions of the key libraries used: ``Python``, ``pandas``, ``NumPy``, ``scikit-learn`` and ``ExplainerDashboard``.

###  2. Data Loading & Exploration
* The **WDBC dataset** is loaded using `load_breast_cancer`.
* The dataset's dimensions, class distribution and a sample of records are examined.

###  3. Exploratory Data Analysis (EDA)
* A bar chart visualizes the distribution of **malignant vs. benign cases**.
* Histograms are generated for key features, including `mean radius`, `mean texture`, `mean perimeter`, `mean area`, and `mean smoothness`.
* A **correlation heatmap** shows the relationships between all features.

###  4. Preprocessing & Splitting
* The data is split into **training (80%) and testing (20%) sets** using a stratified approach to maintain class proportions.
* A **standard scaler** is applied, and the scaled data is converted back into a DataFrame for easier interpretation.

###  5. Model Training
* A **RandomForestClassifier** is trained on the scaled data. The model uses 100 trees and is configured to handle class imbalance (`class_weight='balanced'`).

###  6. Performance Evaluation
* Model performance is assessed using **accuracy** and **ROC AUC** scores on the test set.
* A detailed **classification report** is provided.
* A **confusion matrix** and **ROC curve** are visualized to further analyze the model's performance.

###  7. Global Explainability
* A bar plot displays the **feature importance** for the top 15 predictors, helping to understand which features most influence the model's decisions.
* **Partial dependence plots** are generated for the two most important features to show how they affect the model's output.

###  8. Interactive Explainability Dashboard
* An interactive dashboard is created using `ClassifierExplainer` from **ExplainerDashboard**, with `whatif=True` and `contributions=True` enabled for deeper analysis.
* The dashboard can be accessed by navigating to **`http://127.0.0.1:8050`** when running the project in a JupyterLab environment or a web browser.

## Usage
To begin, follow these two steps:

 **1. Install Dependencies**
 
First, install the necessary libraries by running this command in your terminal:

```pip install -r requirements.txt```

**2. Launch the Notebook**

Next, open the project's Jupyter Notebook:

```jupyter notebook breast_cancer_analysis.ipynb```

Once the notebook is open, you can execute all cells to train the machine learning model and launch the interactive explainer dashboard.

## Acknowledgment

* ``scikit-learn`` for the machine learning library.
* ``pandas`` for data manipulation and analysis.
* ``ExplainerDashboard`` for the interactive dashboard.

