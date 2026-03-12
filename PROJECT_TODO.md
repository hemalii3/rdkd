# RDK Project: Time Series Forecasting with Clustering

## Project To-Do List & Implementation Plan

---

## 📊 **PROJECT OVERVIEW**

**Course:** Recent Developments in KDD (SS 2026)  
**Topic:** Leveraging Clustering for Large-Scale Time Series Forecasting  
**Deadline:** April 30, 2026, 23:59  
**Current Date:** March 12, 2026 ⏰ (~7 weeks remaining)  
**Team Size:** 3-4 students

---

## 📁 **DATASET SUMMARY**

- **Households:** 17,548 unique IDs
- **Training Data:** `sample_23.csv` - 365 days (2023)
- **Testing Data:** `sample_24.csv` - 366 days (2024, leap year)
- **Target Variable:** Daily energy consumption (kWh)
- **Total Data Points:** ~6.4 million observations per year

**⚠️ CRITICAL RULE:** Only use 2023 data for training/clustering. 2024 data is EXCLUSIVELY for evaluation.

---

## 🎯 **GRADING BREAKDOWN**

| Component                     | Points  | Focus Areas                                        |
| ----------------------------- | ------- | -------------------------------------------------- |
| Task 1: Clustering            | 40      | Algorithm choice, k estimation, interpretability   |
| Task 2: Forecasting           | 40      | Model selection, evaluation methodology            |
| Research Diary & Presentation | 20      | Individual contributions, clear communication      |
| **Total**                     | **100** | Novelty, reasoning, explanations > raw performance |

---

## 📋 **PHASE 0: PROJECT SETUP & ORGANIZATION**

### ✅ Setup Tasks

- [ ] **0.1** Form team (3-4 members)
- [ ] **0.2** Setup shared repository (Git)
- [ ] **0.3** Create folder structure:
  ```
  RDK_Project/
  ├── data/                    # Raw CSV files
  ├── notebooks/               # Jupyter notebooks for exploration/reporting
  ├── src/                     # Production Python code
  │   ├── preprocessing/
  │   ├── clustering/
  │   ├── forecasting/
  │   ├── evaluation/
  │   └── utils/
  ├── results/                 # Outputs, figures, metrics
  ├── docs/                    # Report drafts, presentation
  └── research_diaries/        # Individual work logs
  ```
- [ ] **0.4** Setup Python environment (conda/venv) with requirements.txt
- [ ] **0.5** Initialize research diaries for each team member
- [ ] **0.6** Create `.gitignore` (exclude large data files from repo)
- [ ] **0.7** Define code style conventions (follow PEP8)

---

## 📋 **PHASE 1: EXPLORATORY DATA ANALYSIS (EDA)**

**Estimated Time:** Week 1-2 (March 12-25)

### 1.1 Data Loading & Validation

- [ ] **1.1.1** Load both CSV files and verify dimensions
- [ ] **1.1.2** Check for missing values (NaN, zeros, negatives)
- [ ] **1.1.3** Verify household IDs match across 2023 and 2024
- [ ] **1.1.4** Check data types and value ranges (min, max, mean)
- [ ] **1.1.5** Document any data quality issues

### 1.2 Descriptive Statistics

- [ ] **1.2.1** Compute per-household statistics (mean, std, min, max, median)
- [ ] **1.2.2** Identify outlier households (extremely high/low consumption)
- [ ] **1.2.3** Analyze overall distribution of consumption values
- [ ] **1.2.4** Calculate coefficient of variation per household
- [ ] **1.2.5** Detect zero-consumption periods

### 1.3 Temporal Pattern Analysis

- [ ] **1.3.1** Plot sample time series (10-20 random households)
- [ ] **1.3.2** Aggregate consumption by day-of-week
- [ ] **1.3.3** Aggregate consumption by month
- [ ] **1.3.4** Detect seasonality (summer vs winter patterns)
- [ ] **1.3.5** Identify weekly cycles (weekday vs weekend)
- [ ] **1.3.6** Analyze trends (increasing/decreasing consumption over year)

### 1.4 Initial Clustering Insights

- [ ] **1.4.1** Create correlation matrix between random households
- [ ] **1.4.2** Visualize consumption distributions (histograms, box plots)
- [ ] **1.4.3** Identify preliminary consumption groups (visual inspection):
  - Low consumers
  - Medium consumers
  - High consumers
  - Seasonal patterns
  - Stable vs volatile consumers

### 📊 **Deliverable:** EDA Jupyter notebook with visualizations and insights

---

## 📋 **PHASE 2: DATA PREPROCESSING & FEATURE ENGINEERING**

**Estimated Time:** Week 2-3 (March 19 - April 1)

### 2.1 Data Cleaning

- [ ] **2.1.1** Handle missing values (if any):
  - Imputation strategy (forward fill, interpolation, mean)
  - Document approach and impact
- [ ] **2.1.2** Handle outliers:
  - Define outlier threshold (e.g., > 3σ from mean)
  - Cap, remove, or keep (with justification)
- [ ] **2.1.3** Handle zero/negative values (if invalid)
- [ ] **2.1.4** Normalize/standardize data (decide on approach)

### 2.2 Feature Extraction for Clustering

**Goal:** Extract features that capture time series characteristics

- [ ] **2.2.1** Statistical Features (per household):
  - Mean, median, standard deviation
  - Min, max, range
  - Quantiles (25%, 50%, 75%)
  - Skewness, kurtosis
  - Coefficient of variation

- [ ] **2.2.2** Temporal Features:
  - Day-of-week averages (7 features)
  - Weekend vs weekday average ratio
  - Monthly averages (12 features)
  - Quarterly averages (4 features)
  - Trend coefficient (linear regression slope)

- [ ] **2.2.3** Seasonality Features:
  - Summer vs winter consumption ratio
  - Seasonal decomposition components (trend, seasonal, residual)
  - Amplitude of seasonal variation

- [ ] **2.2.4** Variability Features:
  - Rolling window standard deviation (weekly, monthly)
  - Number of peaks/valleys
  - Autocorrelation at lags 1, 7, 30

- [ ] **2.2.5** Shape Features:
  - Time series entropy
  - Spectral features (FFT components)
  - Hurst exponent (long-term memory)

- [ ] **2.2.6** Advanced Features (optional):
  - DTW (Dynamic Time Warping) distance features
  - Catch22 features (time series specific)
  - Wavelet transform features

### 2.3 Feature Selection & Dimensionality Reduction

- [ ] **2.3.1** Calculate feature correlations
- [ ] **2.3.2** Remove highly correlated features (> 0.9)
- [ ] **2.3.3** Apply PCA (Principal Component Analysis) for visualization
- [ ] **2.3.4** Consider t-SNE or UMAP for 2D visualization
- [ ] **2.3.5** Document feature importance/relevance

### 2.4 Data Scaling

- [ ] **2.4.1** Apply StandardScaler or MinMaxScaler to features
- [ ] **2.4.2** Save scaler parameters for consistency

### 📊 **Deliverable:** Clean feature matrix (shape: 17,548 households × N features)

---

## 📋 **PHASE 3: CLUSTERING (TASK 1)**

**Estimated Time:** Week 3-4 (March 26 - April 8)  
**Points:** 40/100

### 3.1 Algorithm Selection & Comparison

Test multiple clustering algorithms:

- [ ] **3.1.1** K-Means Clustering
  - Standard implementation
  - Mini-batch K-Means (for speed)
- [ ] **3.1.2** Hierarchical Clustering
  - Different linkage methods (ward, average, complete)
  - Create dendrograms
- [ ] **3.1.3** DBSCAN (Density-based)
  - Tune eps and min_samples
  - Handles outliers/noise
- [ ] **3.1.4** Gaussian Mixture Models (GMM)
  - Soft clustering alternative
  - Probabilistic approach
- [ ] **3.1.5** Time Series Specific Methods:
  - K-Medoids with DTW distance
  - Time series K-Means (tslearn library)
- [ ] **3.1.6** Optional Advanced Methods:
  - Spectral Clustering
  - HDBSCAN
  - Self-Organizing Maps (SOM)

### 3.2 Optimal K Estimation

**CRITICAL:** No ground truth → must justify K selection

- [ ] **3.2.1** Elbow Method:
  - Plot within-cluster sum of squares (WCSS) vs K
  - Identify "elbow point"
- [ ] **3.2.2** Silhouette Analysis:
  - Compute silhouette score for K = 2 to 20
  - Higher score = better separation
  - Visualize silhouette plots
- [ ] **3.2.3** Davies-Bouldin Index:
  - Lower is better
  - Measures cluster separation
- [ ] **3.2.4** Calinski-Harabasz Index:
  - Higher is better
  - Ratio of between-cluster to within-cluster variance
- [ ] **3.2.5** Gap Statistic:
  - Compare WCSS to random reference distribution
- [ ] **3.2.6** Dendrogram Analysis (for hierarchical):
  - Visual inspection of optimal cut height
- [ ] **3.2.7** Domain-Driven Heuristics:
  - Business logic (e.g., small/medium/large consumers)
  - Computational feasibility for forecasting
- [ ] **3.2.8** Stability Analysis:
  - Bootstrap resampling
  - Check if clusters remain stable

### 3.3 Final Clustering

- [ ] **3.3.1** Select optimal algorithm and K (document reasoning)
- [ ] **3.3.2** Perform final clustering on 2023 data
- [ ] **3.3.3** Assign cluster labels to all 17,548 households
- [ ] **3.3.4** Save cluster assignments (household_id → cluster_id mapping)

### 3.4 Cluster Interpretation & Validation

- [ ] **3.4.1** Compute cluster centroids/prototypes
- [ ] **3.4.2** Analyze cluster characteristics:
  - Size (number of households per cluster)
  - Average consumption per cluster
  - Temporal patterns per cluster (plot mean time series)
  - Feature distributions per cluster
- [ ] **3.4.3** Visualize clusters:
  - 2D scatter (PCA/t-SNE) colored by cluster
  - Heatmaps of cluster centroids
  - Time series plots by cluster
- [ ] **3.4.4** Cluster naming/labeling (interpretability):
  - Example: "High Winter Consumers", "Stable Low Consumers", etc.
- [ ] **3.4.5** Internal validation metrics:
  - Silhouette score
  - Davies-Bouldin index
  - Calinski-Harabasz index
- [ ] **3.4.6** Check for imbalanced clusters (very small/large)

### 📊 **Deliverable:**

- Cluster assignments file
- Clustering analysis notebook
- Visualizations and interpretation report

---

## 📋 **PHASE 4: FORECASTING (TASK 2)**

**Estimated Time:** Week 4-5 (April 2-15)  
**Points:** 40/100

### 4.1 Forecasting Strategy Design

#### 4.1.1 Cluster-Level Forecasting

- [ ] Train one model per cluster
- [ ] Each model learns from all households in that cluster
- [ ] Generate 366 predictions per household for 2024

#### 4.1.2 Global Baseline Forecasting

- [ ] Train single model on entire 2023 dataset
- [ ] Generate 366 predictions per household for 2024

### 4.2 Model Selection & Comparison

Test multiple forecasting models:

#### Statistical Models

- [ ] **4.2.1** Naive Baseline:
  - Last value carry-forward
  - Seasonal naive (lag 7, lag 365)
- [ ] **4.2.2** ARIMA / SARIMA:
  - Auto-ARIMA for parameter tuning
  - Per-cluster or global
- [ ] **4.2.3** Exponential Smoothing (ETS):
  - Holt-Winters method
  - Handles trend and seasonality

#### Machine Learning Models

- [ ] **4.2.4** Linear Regression:
  - With lag features (past 7, 14, 30 days)
  - Seasonal features (day-of-week, month)
- [ ] **4.2.5** Random Forest Regressor:
  - Robust to outliers
  - Feature importance analysis
- [ ] **4.2.6** Gradient Boosting (XGBoost, LightGBM):
  - State-of-the-art tabular performance
  - Hyperparameter tuning
- [ ] **4.2.7** Support Vector Regression (SVR):
  - Kernel-based approach

#### Deep Learning Models

- [ ] **4.2.8** LSTM (Long Short-Term Memory):
  - Sequence-to-sequence
  - Handles long-term dependencies
- [ ] **4.2.9** GRU (Gated Recurrent Unit):
  - Simpler than LSTM, faster training
- [ ] **4.2.10** Temporal Convolutional Network (TCN):
  - 1D CNNs for time series
- [ ] **4.2.11** Transformer-based Models:
  - Temporal Fusion Transformer (TFT)
  - Informer

#### Specialized Time Series Models

- [ ] **4.2.12** Prophet (Facebook):
  - Handles seasonality, holidays
  - Fast and interpretable
- [ ] **4.2.13** N-BEATS (Neural Basis Expansion):
  - Pure deep learning, no feature engineering

### 4.3 Feature Engineering for Forecasting

- [ ] **4.3.1** Lag Features:
  - Past 1, 2, 3, 7, 14, 30 days
- [ ] **4.3.2** Rolling Statistics:
  - 7-day moving average
  - 30-day moving average
  - Rolling std, min, max
- [ ] **4.3.3** Temporal Features:
  - Day of week (one-hot encoded)
  - Month (one-hot encoded)
  - Day of year
  - Week of year
  - Is weekend (binary)
- [ ] **4.3.4** Seasonal Features:
  - Season (winter/spring/summer/fall)
  - Cyclical encoding (sin/cos transforms)
- [ ] **4.3.5** Cluster Membership (for global model):
  - Cluster ID as categorical feature

### 4.4 Training & Validation Strategy

- [ ] **4.4.1** Define train/validation split on 2023 data:
  - Option A: Temporal split (first 10 months train, last 2 months validation)
  - Option B: Walk-forward validation
- [ ] **4.4.2** Hyperparameter tuning:
  - Grid search or Random search
  - Cross-validation strategy
- [ ] **4.4.3** Train cluster-level models:
  - For each cluster k:
    - Combine time series of all households in cluster k
    - Train model(k) on this data
    - Save model(k)
- [ ] **4.4.4** Train global baseline model:
  - Combine time series of ALL households
  - Train single global model
  - Save model

### 4.5 Prediction Generation

- [ ] **4.5.1** Generate 366-day forecasts for each household:
  - Cluster-based: Use model(k) where household ∈ cluster k
  - Global: Use global model for all households
- [ ] **4.5.2** Handle prediction horizon:
  - Iterative forecasting (one-step ahead, rolling)
  - OR Direct forecasting (multi-step output)
- [ ] **4.5.3** Save predictions:
  - Format: household_id × 366 days (matching 2024 structure)

### 📊 **Deliverable:**

- Trained models (cluster-based and global)
- Prediction files for 2024 (both approaches)

---

## 📋 **PHASE 5: EVALUATION**

**Estimated Time:** Week 5 (April 9-15)

### 5.1 Evaluation Metrics

- [ ] **5.1.1** Mean Absolute Error (MAE) - PRIMARY METRIC:
  - For each household: MAE = mean(|predicted - actual|) over 366 days
  - Final metric: Average MAE across all 17,548 households
- [ ] **5.1.2** Additional metrics (for deeper analysis):
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - R² Score
  - Median Absolute Error (robust to outliers)

### 5.2 Household-Level Evaluation

- [ ] **5.2.1** Compute MAE for each household (cluster-based)
- [ ] **5.2.2** Compute MAE for each household (global baseline)
- [ ] **5.2.3** Visualize MAE distributions (histograms, box plots)
- [ ] **5.2.4** Identify best/worst performing households

### 5.3 Cluster-Level Evaluation

- [ ] **5.3.1** Compute average MAE per cluster
- [ ] **5.3.2** Analyze which clusters forecast well/poorly
- [ ] **5.3.3** Investigate why certain clusters fail (heterogeneity?)

### 5.4 Comparative Analysis

- [ ] **5.4.1** Compare cluster-based vs global approach:
  - Paired t-test or Wilcoxon signed-rank test
  - Effect size calculation
- [ ] **5.4.2** Analyze when clustering helps:
  - Which household types benefit from clustering?
- [ ] **5.4.3** Error analysis:
  - Plot prediction errors over time (temporal patterns)
  - Correlation between error and consumption level

### 5.5 Visualization

- [ ] **5.5.1** Sample prediction plots (10-20 households):
  - True vs predicted consumption over 366 days
  - Separate plots for cluster-based and global
- [ ] **5.5.2** Aggregate results plots:
  - Bar chart: Average MAE by cluster vs global
  - Scatter plot: Household-level MAE (cluster vs global)
- [ ] **5.5.3** Residual analysis:
  - Histogram of residuals
  - Q-Q plot
  - Residuals over time

### 📊 **Deliverable:**

- Evaluation report with metrics
- Comparison analysis
- Visualizations

---

## 📋 **PHASE 6: REPORTING & DOCUMENTATION**

**Estimated Time:** Week 6 (April 16-22)

### 6.1 Code Documentation

- [ ] **6.1.1** Add docstrings to all functions (Google/NumPy style)
- [ ] **6.1.2** Add inline comments for complex logic
- [ ] **6.1.3** Create README.md with:
  - Project overview
  - Setup instructions (environment, dependencies)
  - How to run the code (step-by-step)
  - File structure explanation
- [ ] **6.1.4** Create requirements.txt or environment.yml
- [ ] **6.1.5** Add example usage in README

### 6.2 Written Report (PDF/HTML)

**Structure:**

- [ ] **6.2.1** Introduction:
  - Problem statement
  - Dataset description
  - Objectives
- [ ] **6.2.2** Methodology:
  - Data preprocessing approach
  - Feature engineering details
  - Clustering approach (algorithm, K selection, rationale)
  - Forecasting approach (models, parameters)
- [ ] **6.2.3** Results:
  - Clustering results (interpretation, visualization)
  - Forecasting results (MAE, comparisons)
  - Tables and figures
- [ ] **6.2.4** Discussion:
  - When does clustering help?
  - Limitations of approach
  - Future improvements
- [ ] **6.2.5** Conclusion:
  - Key findings
  - Recommendations
- [ ] **6.2.6** Appendix:
  - Additional visualizations
  - Hyperparameter tuning details
  - Cluster characteristics tables
- [ ] **6.2.7** References:
  - Cite papers, libraries, external sources

### 6.3 Research Diaries

- [ ] **6.3.1** Each team member finalizes their diary:
  - Date, task, hours worked
  - Contributions (coding, analysis, writing)
  - Challenges faced
- [ ] **6.3.2** Ensure all diaries are complete and honest

### 📊 **Deliverable:**

- PDF/HTML report
- Complete code with documentation
- Research diaries

---

## 📋 **PHASE 7: PRESENTATION PREPARATION**

**Estimated Time:** Week 6-7 (April 23-29)  
**Duration:** 10 minutes

### 7.1 Slide Preparation

- [ ] **7.1.1** Slide 1: Title + Team members
- [ ] **7.1.2** Slide 2: Problem & Dataset overview
- [ ] **7.1.3** Slide 3-4: Approach (clustering + forecasting)
- [ ] **7.1.4** Slide 5-6: Results (key visualizations, metrics)
- [ ] **7.1.5** Slide 7: Insights & Discussion
- [ ] **7.1.6** Slide 8: Conclusion & Future work
- [ ] **7.1.7** Optional: Backup slides (detailed analysis)

### 7.2 Presentation Practice

- [ ] **7.2.1** Allocate speaking time per team member
- [ ] **7.2.2** Rehearse presentation (aim for 8-9 minutes, leave time for Q&A)
- [ ] **7.2.3** Prepare answers to potential questions:
  - Why this value of K?
  - Why this forecasting model?
  - What are limitations?
  - How to handle new households (cold-start)?

### 📊 **Deliverable:**

- Presentation slides (PDF/PowerPoint)
- Rehearsed 10-minute talk

---

## 📋 **PHASE 8: FINAL SUBMISSION**

**Deadline:** April 30, 2026, 23:59 ⏰

### 8.1 Prepare Submission Package

- [ ] **8.1.1** Create zip archive: `team_X.zip` (replace X with team number)
- [ ] **8.1.2** Include:
  - `report.pdf` or `report.html`
  - `presentation.pdf` or `presentation.pptx`
  - `code/` folder (all Python files)
  - `notebooks/` folder (Jupyter notebooks)
  - `README.md` (how to run)
  - `requirements.txt` or `environment.yml`
  - `research_diaries/` folder (all team members)
  - `results/` folder (key outputs, figures)
- [ ] **8.1.3** Verify zip structure:
  ```
  team_X.zip
  ├── README.md
  ├── report.pdf
  ├── presentation.pdf
  ├── requirements.txt
  ├── code/
  │   ├── preprocessing.py
  │   ├── clustering.py
  │   ├── forecasting.py
  │   └── evaluation.py
  ├── notebooks/
  │   ├── 01_EDA.ipynb
  │   ├── 02_Clustering.ipynb
  │   └── 03_Forecasting.ipynb
  ├── research_diaries/
  │   ├── diary_member1.pdf
  │   ├── diary_member2.pdf
  │   └── ...
  └── results/
      ├── figures/
      └── metrics/
  ```

### 8.2 Quality Checks

- [ ] **8.2.1** Run all code from scratch to ensure reproducibility
- [ ] **8.2.2** Check all file paths are relative (not absolute)
- [ ] **8.2.3** Proofread report (spelling, grammar)
- [ ] **8.2.4** Verify all citations are included
- [ ] **8.2.5** Check all figures have captions and are referenced
- [ ] **8.2.6** Ensure code follows PEP8 style guide

### 8.3 Upload to Moodle

- [ ] **8.3.1** Upload `team_X.zip` to Moodle **BEFORE 23:59 on April 30**
- [ ] **8.3.2** Verify upload was successful (download and check)
- [ ] **8.3.3** Submit on time (NO deadline extensions possible)

---

## 🔧 **RECOMMENDED TOOLS & LIBRARIES**

### Python Libraries

**Data Handling:**

- `pandas` - Data manipulation
- `numpy` - Numerical computing

**Visualization:**

- `matplotlib` - Basic plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive plots

**Clustering:**

- `scikit-learn` - K-Means, DBSCAN, Hierarchical, GMM
- `tslearn` - Time series clustering with DTW
- `hdbscan` - Density-based clustering

**Forecasting:**

- `statsmodels` - ARIMA, SARIMA, ETS
- `prophet` - Facebook Prophet
- `xgboost`, `lightgbm` - Gradient boosting
- `sklearn.ensemble` - Random Forest
- `tensorflow` or `pytorch` - Deep learning (LSTM, GRU)
- `darts` - Time series forecasting library

**Evaluation:**

- `scikit-learn.metrics` - MAE, RMSE, etc.
- `scipy.stats` - Statistical tests

---

## 📝 **RESEARCH DIARY TEMPLATE**

Each team member should maintain a diary with:

| Date       | Task                     | Hours | Details                                          | Challenges                        |
| ---------- | ------------------------ | ----- | ------------------------------------------------ | --------------------------------- |
| 2026-03-12 | Project setup, EDA start | 3h    | Loaded data, checked dimensions, plotted samples | Large file size, memory issues    |
| 2026-03-15 | Feature extraction       | 4h    | Implemented statistical & temporal features      | Decided on which features to keep |
| ...        | ...                      | ...   | ...                                              | ...                               |

**Include:**

- Date and duration
- Specific tasks performed
- Code contributions (which files/functions)
- Analysis contributions (which insights)
- Challenges faced and how you solved them
- Collaboration with team members

---

## ⚠️ **CRITICAL SUCCESS FACTORS**

### Do's ✅

1. **Justify all decisions** - Why this K? Why this model? (more important than performance)
2. **Interpretability over complexity** - Explain cluster meanings clearly
3. **Rigorous evaluation** - Use ONLY 2024 data for testing
4. **Clean code** - Well-documented, readable, PEP8 compliant
5. **Visualize everything** - Clusters, predictions, errors
6. **Cite sources** - Papers, libraries, techniques
7. **Honest research diaries** - Actual contributions and hours
8. **Early submission** - Don't wait until 23:55

### Don'ts ❌

1. **DO NOT use 2024 data for training/clustering**
2. **DO NOT overfit** - Validate properly on 2023
3. **DO NOT ignore data quality** - Check for anomalies
4. **DO NOT choose K arbitrarily** - Must justify with metrics
5. **DO NOT submit late** - No extensions allowed
6. **DO NOT plagiarize** - Cite all external code/ideas
7. **DO NOT overcomplicate** - Simple + explainable > complex black box

---

## 🎯 **MILESTONES & DEADLINES**

| Week         | Dates            | Milestone                    | Deliverable         |
| ------------ | ---------------- | ---------------------------- | ------------------- |
| 1            | Mar 12-18        | Project setup + EDA complete | EDA notebook        |
| 2            | Mar 19-25        | Feature engineering complete | Feature matrix      |
| 3            | Mar 26-Apr 1     | Clustering complete          | Cluster assignments |
| 4            | Apr 2-8          | Forecasting models trained   | Predictions file    |
| 5            | Apr 9-15         | Evaluation complete          | Results analysis    |
| 6            | Apr 16-22        | Report drafted               | First draft report  |
| 7            | Apr 23-29        | Final polish + presentation  | Final submission    |
| **Deadline** | **Apr 30 23:59** | **SUBMIT TO MOODLE**         | **team_X.zip**      |

---

## 💡 **ADVANCED TIPS**

### Cold-Start Problem (mentioned in project)

If a new household starts:

1. Extract features from their limited data
2. Compute distance to cluster centroids
3. Assign to nearest cluster
4. Apply that cluster's forecasting model

### Computational Efficiency

- 17,548 households is large-scale
- Use parallel processing (joblib, multiprocessing)
- Consider sampling for rapid prototyping
- Use efficient algorithms (Mini-batch K-Means)

### Novelty Points

- Hybrid clustering (combine statistical + DTW-based)
- Ensemble forecasting (combine multiple models)
- Hierarchical clustering with different granularities
- Attention mechanisms for household importance weighting
- Transfer learning from similar time series datasets

---

## 📚 **RECOMMENDED READING**

### Papers

- "Clustering of Time Series Based on Shape Characteristics" (survey)
- "Time Series Clustering – A Review" (Aghabozorgi et al.)
- "M4 Competition Results" (forecasting benchmarks)

### Books

- "Forecasting: Principles and Practice" (Hyndman & Athanasopoulos) - Free online
- "Introduction to Time Series Forecasting with Python" (Brownlee)

### Online Resources

- Scikit-learn Time Series documentation
- TSlearn documentation
- Prophet documentation
- Kaggle time series competitions (M5, web traffic, energy)

---

## 🤝 **TEAM COLLABORATION**

### Communication

- [ ] Setup team communication channel (Slack, Discord, WhatsApp)
- [ ] Schedule weekly sync meetings (1-2 hours)
- [ ] Use Google Docs / Overleaf for collaborative report writing

### Version Control

- [ ] Use Git for code management
- [ ] Meaningful commit messages
- [ ] Code review before merging
- [ ] Branch strategy (main, dev, feature branches)

### Task Allocation

**Example split for 4 members:**

- **Member 1:** EDA + Data Preprocessing
- **Member 2:** Clustering (algorithm comparison, K selection)
- **Member 3:** Forecasting (model development, training)
- **Member 4:** Evaluation + Report writing

Everyone contributes to presentation and final polish.

---

## ✅ **FINAL CHECKLIST BEFORE SUBMISSION**

- [ ] Code runs without errors on clean environment
- [ ] All results are reproducible
- [ ] Report is complete (intro, methods, results, discussion, conclusion)
- [ ] All figures have captions and are referenced in text
- [ ] All external sources are cited
- [ ] Research diaries are complete for all members
- [ ] Presentation is ready (10 minutes, rehearsed)
- [ ] Zip file is correctly named: `team_X.zip`
- [ ] Zip contains all required files
- [ ] Uploaded to Moodle before deadline
- [ ] Upload verified (downloaded and checked)

---

## 🚀 **GOOD LUCK!**

**Remember:** The grade is based on:

- **Novelty** - Creative approaches
- **Interpretability** - Clear explanations
- **Reasoning** - Justified decisions
- **Communication** - Well-written report and presentation

Performance (MAE) is important but **NOT the only factor**!

---

**Last Updated:** March 12, 2026  
**Status:** Project planning phase  
**Next Action:** Start Phase 0 (Setup) and Phase 1 (EDA)
