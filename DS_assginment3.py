'''Investigation A: Do Bats Perceive Rats as Predators?

This notebook addresses Objective 1 (Assessment 2) of the HIT140 Foundations of Data Science project.

We investigate whether bats perceive rats only as food competitors or also as potential predators.  
If rats are considered a predation risk, bats should show more avoidance behaviour or increased vigilance during foraging.

We use two datasets:
- dataset1.csv → Bat landings & behaviours (risk, reward, season, etc.)  
- dataset2.csv → Rat arrivals & bat activity (arrivals, food availability, etc.)

We will perform both:
1. Descriptive analysis (counts, trends, plots)  
2. Inferential analysis (statistical tests: chi-square, correlation)
'''


'''Importing libraries
    1. Pandas to process the provided data in dataset1 and dataset2
    2. matplotlib for plotting and visualization
    3. seaborn is a highlevel interface built on top of matplotlib which can work directly on pandas df
    4. scipy is used to perform scientific calculation such as correlation, chi square test etc.
'''    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuring the specific style i.e graph background is white, muted colors and bigger font size
sns.set(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)

# Loading the data sets using the pandas library
dataset1 = pd.read_csv("dataset1.csv")
dataset2 = pd.read_csv("dataset2.csv")

# Printing the shape of the provided datasets
print("Loaded datasets:")
print("Dataset1 shape:", dataset1.shape)
print("Dataset2 shape:", dataset2.shape)

# Previewing the data sets for futhur cleaning and processing
print("Dataset1 (Bat Landings):")
# head function returns the first 5 rows of the data frame
print(dataset1.head())
# provides the detail information of the columns in the data frame
print(dataset1.info())

# Previewing the data sets for futhur cleaning and processing same as dataset 1
print("\nDataset2 (Rat Arrivals):")
print(dataset2.head())
print(dataset2.info())

# Converting the columns name to lower case for standardising purpose
dataset1.columns = [c.strip().lower() for c in dataset1.columns]
dataset2.columns = [c.strip().lower() for c in dataset2.columns]

# printing the columns of the dataset as a list
print("\nDataset1 columns:", dataset1.columns.tolist())
print("Dataset2 columns:", dataset2.columns.tolist())