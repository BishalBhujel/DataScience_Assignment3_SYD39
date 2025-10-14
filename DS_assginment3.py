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
import numpy as np
from scipy.stats import chi2_contingency, pearsonr, ttest_ind
from statsmodels.stats.proportion import proportions_ztest

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

# Defining the column names for better readability
risk_col = 'risk'
reward_col = 'reward'
season_col = 'season'
rat_arrivals_col = 'rat_arrival_number'
bat_landings_col = 'bat_landing_number'
food_col = 'food_availability'
time_col_d1 = 'start_time'
time_col_d2 = 'time'

# Checking for missing values in dataset1 and converting it to numeric
for col in [risk_col, reward_col]:
    if col in dataset1.columns:
        dataset1[col] = pd.to_numeric(dataset1[col], errors='coerce').astype('Int64')

# Checking for missing values in dataset2, and converting it to numeric
for col in [rat_arrivals_col, bat_landings_col, food_col]:
    if col in dataset2.columns:
        dataset2[col] = pd.to_numeric(dataset2[col], errors='coerce')

# Clean season labels and converting the season column data into lowercase
if season_col in dataset1.columns:
    dataset1[season_col] = dataset1[season_col].astype(str).str.strip().str.lower()

# Parsing the date time from the date time column present in both datasets
for c, df in [(time_col_d1, dataset1), (time_col_d2, dataset2)]:
    if c in df.columns:
        try:
            df[c] = pd.to_datetime(df[c])
        except:
            pass

# dropping the NA values for risk column in dataset 1 and printing the dropped rows number
if risk_col in dataset1.columns:
    before = dataset1.shape
    dataset1 = dataset1.dropna(subset=[risk_col])
    print(f"\nDropped rows with missing '{risk_col}' in dataset1: {before} -> {dataset1.shape}")

# dropping the NA values for rat_arrival_number and bat_arrival_number columns in dataset 2 and printing the dropped rows number
if (rat_arrivals_col in dataset2.columns) and (bat_landings_col in dataset2.columns):
    before = dataset2.shape
    dataset2 = dataset2.dropna(subset=[rat_arrivals_col, bat_landings_col])
    print(f"Dropped rows with missing '{rat_arrivals_col}'/'{bat_landings_col}' in dataset2: {before} -> {dataset2.shape}")

# Peinting the shape of the datasets after dropping the NA rows
print("\nAfter cleaning - shapes:")
print(" - dataset1:", dataset1.shape)
print(" - dataset2:", dataset2.shape)

# Performing feature engineering for furthur analysis

# Performing the analysis to find out rat intensity and presence
if rat_arrivals_col in dataset2.columns:
    dataset2['rat_present'] = (dataset2[rat_arrivals_col] > 0).astype(int)
    def rat_intensity(x):
        if pd.isna(x): return np.nan
        if x == 0: return 0
        if x <= 2: return 1
        if x <= 5: return 2
        return 3
    dataset2['rat_intensity'] = dataset2[rat_arrivals_col].apply(rat_intensity)

# Calculating Season inference if missing
if season_col not in dataset1.columns and 'month' in dataset1.columns:
    dataset1['season'] = dataset1['month'].map({
        12:'summer',1:'summer',2:'summer',3:'autumn',4:'autumn',5:'autumn',
        6:'winter',7:'winter',8:'winter',9:'spring',10:'spring',11:'spring'
    })

if season_col not in dataset2.columns and 'month' in dataset2.columns:
    dataset2['season'] = dataset2['month'].map({
        12:'summer',1:'summer',2:'summer',3:'autumn',4:'autumn',5:'autumn',
        6:'winter',7:'winter',8:'winter',9:'spring',10:'spring',11:'spring'
    })

# Merge dataset1 with interval info from dataset2 if time available
def floor_to_30min(dt):
    return dt.dt.floor('30T')

if (time_col_d1 in dataset1.columns) and (time_col_d2 in dataset2.columns):
    dataset1['interval'] = floor_to_30min(dataset1[time_col_d1])
    dataset2['interval'] = floor_to_30min(dataset2[time_col_d2])
    merged = dataset1.merge(dataset2[['interval','rat_present',rat_arrivals_col,
                                      'rat_intensity',food_col,bat_landings_col]],
                            on='interval', how='left')
else:
    merged = dataset1.copy()
    merged['rat_present'] = np.nan

print("Merged dataset shape:", merged.shape)


# Performing and plotting graph for Investigation A: Risk-taking vs rat presence
if 'risk' in merged.columns and 'rat_present' in merged.columns:
    temp = merged.dropna(subset=['risk','rat_present'])
    if len(temp) > 0:
        rat_risk_ct = pd.crosstab(temp['rat_present'], temp['risk'])
        print("Contingency Table (Rat Presence x Risk):\n", rat_risk_ct)

        prop_by_rat = temp.groupby('rat_present')['risk'].mean().reset_index()
        if not prop_by_rat.empty:
            sns.barplot(x='rat_present', y='risk', data=prop_by_rat)
            plt.xticks([0,1], ['No Rats', 'Rats Present'])
            plt.ylabel('Proportion Risk-taking')
            plt.title('Risk-taking vs Rat Presence')
            plt.show()

# Performing and plotting graph for Investigation A: Rat arrivals vs Bat landings
if rat_arrivals_col in dataset2.columns and bat_landings_col in dataset2.columns:
    temp = dataset2.dropna(subset=[rat_arrivals_col, bat_landings_col])
    if len(temp) > 0:
        sns.scatterplot(x=rat_arrivals_col, y=bat_landings_col, data=temp, alpha=0.6)
        plt.title('Rat Arrivals vs Bat Landings')
        plt.show()

# Performing and plotting graph for Investigation B: Risk-taking by season
if 'season' in merged.columns and 'risk' in merged.columns:
    temp = merged.dropna(subset=['season','risk'])
    if len(temp) > 0:
        sns.barplot(x='season', y='risk', data=temp, estimator=np.mean,
                    order=sorted(temp['season'].unique()))
        plt.title('Risk-taking by Season')
        plt.ylabel('Proportion Risk-taking')
        plt.show()

# Performing and plotting graph for Investigation B: Bat landings by season
if 'season' in dataset2.columns and 'bat_landing_number' in dataset2.columns:
    temp = dataset2.dropna(subset=['season','bat_landing_number'])
    if len(temp) > 0:
        sns.boxplot(x='season', y='bat_landing_number', data=temp,
                    order=sorted(temp['season'].unique()))
        plt.title('Bat Landings by Season')
        plt.show()

# Investigation A: Performing Chi-square test
if 'rat_present' in merged.columns and 'risk' in merged.columns:
    temp = merged.dropna(subset=['rat_present','risk'])
    if len(temp) > 0:
        contingency = pd.crosstab(temp['rat_present'], temp['risk'])
        if contingency.shape[0] > 0 and contingency.shape[1] > 0:
            chi2, p, dof, expected = chi2_contingency(contingency.fillna(0))
            print(f"Chi-square: chi2={chi2:.3f}, dof={dof}, p={p:.4f}")

# Investigation A: Performing Proportions z-test
if 'rat_present' in merged.columns and 'risk' in merged.columns:
    temp = merged.dropna(subset=['rat_present','risk'])
    grouped = temp.groupby('rat_present')['risk'].agg(['sum','count']).reset_index()
    if grouped.shape[0] == 2:
        stat, pval = proportions_ztest(grouped['sum'].values, grouped['count'].values)
        print(f"Proportions z-test: stat={stat:.3f}, p={pval:.4f}")

# Investigation A: Performing Correlation Test
if rat_arrivals_col in dataset2.columns and bat_landings_col in dataset2.columns:
    temp = dataset2.dropna(subset=[rat_arrivals_col, bat_landings_col])
    if len(temp) > 1:
        r, p = pearsonr(temp[rat_arrivals_col], temp[bat_landings_col])
        print(f"Pearson correlation: r={r:.3f}, p={p:.4f}")

# Investigation B: Performing Seasonal risk-taking (Winter vs Spring)
if 'season' in merged.columns and 'risk' in merged.columns:
    temp = merged.dropna(subset=['season','risk'])
    if {'winter','spring'}.issubset(set(temp['season'].unique())):
        winter = temp[temp['season']=='winter']['risk']
        spring = temp[temp['season']=='spring']['risk']
        if len(winter) > 0 and len(spring) > 0:
            stat, pval = proportions_ztest([winter.sum(), spring.sum()],
                                           [len(winter), len(spring)])
            print(f"Winter vs Spring risk-taking z-test: stat={stat:.3f}, p={pval:.4f}")

# Investigation B: Performing Seasonal bat landings
if 'season' in dataset2.columns and 'bat_landing_number' in dataset2.columns:
    temp = dataset2.dropna(subset=['season','bat_landing_number'])
    if {'winter','spring'}.issubset(set(temp['season'].unique())):
        w = temp[temp['season']=='winter']['bat_landing_number']
        s = temp[temp['season']=='spring']['bat_landing_number']
        if len(w) > 1 and len(s) > 1:
            tstat, tp = ttest_ind(w, s, equal_var=False, nan_policy='omit')
            print(f"Winter vs Spring bat landings t-test: t={tstat:.3f}, p={tp:.4f}")