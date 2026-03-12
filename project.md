University of Vienna Vienna,  
Faculty of Computer Science
Recent developments in KDD
SS 2026
Leveraging Clustering for Large-Scale Time Series
Forecasting
Project work
Submission until: April 30, 2026, 23:59
General Remarks:
● This project should be conducted in teams of 3 to 4 students.
● The final deadline for the submission is at 23:59 o’clock on 30.04.2026. Please
upload your solutions to Moodle. No deadline extension is possible.
● To answer the questions, please use Python. Your code must be well-documented
and readable; see the Style Guide for Python1 for common style conventions. You
may use Jupyter notebooks for your report in which you explain and visualize your
results, but the main code base should be in separate Python files.
● Upload your solutions as a zip archive with the following naming scheme: team X.zip,
where X is your team’s number. The archive should contain your code as well as a
PDF or HTML report describing your approach, including your assumptions, results
and a description of how to compile and run your code. Additionally, you should
prepare a 10-minute presentation outlining your approach and results.
● Every member of the group should keep a “Research Diary” to report contributions
and working hours. It should be submitted together with the project work.  
● Mark and cite the external sources you are using in the code and report.
● You can use the designated forum to ask questions about the project or contact the
tutors of the course.
Time series data is a sequence of data points recorded at specific, successive time
intervals (e.g., hourly electricity usage for every house in a city) and is one of the most
prevalent data structures in modern analytics.  
Unlike static datasets, time series are defined by their temporal dependencies, in which the
ordering of points encapsulates critical information about trends, seasonality, and cyclic
behaviour. However, as the scale of data grows from dozens to millions of individual series,
traditional "one-model-per-series" forecasting approaches often become computationally
untenable and statistically fragile.
This project explores clustering as a strategic preprocessing step to manage this
complexity. By grouping time series that exhibit similar structural characteristics—such as
shared seasonal peaks or comparable growth trajectories—we can transition from a
fragmented modelling landscape to a more unified framework.
Clustering enhances forecasting through several key mechanisms:
● Pattern Recognition: It identifies "representative" behaviours, allowing you to build
robust models for a few clusters rather than thousands of individuals.
● Dimensionality Reduction and Scalability: By partitioning the dataset into distinct
groups, you simplify the forecasting task.
● Cold-Start Solutions: If a new data stream starts (e.g., a new customer/house), you
can assign it to an existing cluster to immediately apply an accurate forecasting logic.
Dataset:
The dataset contains daily energy consumption for two years: 2023 (training year) and 2024
(testing year). All data preprocessing, clustering and model training must be performed using
the 2023 dataset only.
The 2024 dataset must be used only for evaluation. No information from 2024 may be used
during clustering or model training.
You can download the dataset from
https://ucloud.univie.ac.at/index.php/s/o5295C8mQo6Jg6m
There are two files:
● sample_23.csv: which contains the daily energy consumption of 2023.
● sample_24.csv: which contains the daily energy consumption of 2024.
Please note that the first column in both files is an ID column representing the household ID,
and it is the same for both 2023 and 2024. Additionally, note that 2024 is a leap year, so it
has 366 days.
Task 1: Clustering
In this part, your goal is to do clustering for the 2023 dataset. You can design your clustering
pipeline as you see fit by performing the necessary data understanding, preprocessing,
cleaning, and comparing clustering algorithms, etc.  
Since there is no ground truth information, you need to estimate the true number of clusters
(k) and explain why and how you chose that number specifically.
Task 2: Forecasting
Perform forecasting at 2 different levels:
● Cluster level: based on the clusters obtained in Task 1, train one forecasting model
per cluster using the 2023 data of the households assigned to that cluster. Together
these models must produce 366 daily predictions for each individual household in 2024.  
● Dataset level: as a baseline, train a single global forecasting model on the entire
2023 dataset without clustering. Predict each household's daily energy consumption
for 2024 using only the 2023 dataset. For every household, you must generate 366
daily forecasts.
Evaluation. Use the 2024 data exclusively as ground truth.  
Perform evaluation at household level:
● for each household, compare the 366 predicted daily values with the corresponding
observed daily consumption values. Compute the Mean Absolute Error (MAE) over
the 366 daily predictions for each household. The final reported performance must be
the average MAE across all households. Both the global model and the cluster-based
models must be evaluated using this procedure.
Grading:
Task
Clustering
Forecasting
Points
40
40
Research Diary and Presentation
20
The grade is not solely based on prediction performance; i.e., better performance does not
immediately mean a better grade. The grade is based on, but not limited to, aspects such
as novelty, interpretability, explanations, and the reasoning behind the chosen methods.
