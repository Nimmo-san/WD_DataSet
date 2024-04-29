import wooldridge as wd
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from linearmodels.panel import PanelOLS
from statsmodels.sandbox.regression.gmm import IV2SLS

df = wd.data('wagepan')


""" Clean and prepare your data here """
# Detecting and handling outliers 
df['lwage_z'] = zscore(df['lwage'])
df = df[(df['lwage_z'] > -3) & (df['lwage_z'] < 3)]  # Keep only within 3 SD

# Calculate Q1, Q3, and the IQR
Q1 = df['lwage'].quantile(0.25)
Q3 = df['lwage'].quantile(0.75)
IQR = Q3 - Q1

# Define the bounds for the outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
df = df[(df['lwage'] >= lower_bound) & (df['lwage'] <= upper_bound)]

print(lower_bound)
print(upper_bound)

# Dropping the z-score column after outliers have been handled
df.drop('lwage_z', axis=1, inplace=True)


#print(df.head())

# EDA

# Summary statistics for key variables
print(df[['lwage', 'exper', 'educ', 'union']].describe())

# Proportion of categorical variables
print(df[['union', 'black', 'married']].apply(pd.Series.value_counts, normalize=True))

# Histogram of lwage
sns.histplot(df['lwage'], kde=True)
plt.title('Distribution of Log(Wages)')
plt.show()

# Boxplot of lwage by union membership
sns.boxplot(x='union', y='lwage', data=df)
plt.title('Log(Wages) by Union Membership')
plt.show()

# Correlation matrix
corr_matrix = df[['lwage', 'exper', 'educ', 'union']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Time series plot
df['union_status'] = df['union'].apply(lambda x: 'Union' if x == 1 else 'Non-Union')

# Group by time and union status, then calculate mean wages
grouped = df.groupby(['year', 'union_status'])['lwage'].mean().reset_index()

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(data=grouped, x='year', y='lwage', hue='union_status')
plt.title('Average Log Wages Over Time by Union Status')
plt.xlabel('Year')
plt.ylabel('Average Log Wage')
plt.show()

# Scatter Plot showing how wages change with experience
sns.lmplot(x='exper', y='lwage', hue='union_status', data=df, aspect=1.5, scatter_kws={'alpha':0.5})
plt.title('Impact of Experience on Wages by Union Status')
plt.xlabel('Years of Experience')
plt.ylabel('Log of Wages')
plt.show()

# lwage vs educ
plt.figure(figsize=(10, 6))
sns.regplot(x='hours', y='lwage', data=df, ci=None, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Log(Wages) vs. Hours')
plt.xlabel('Years of Education')
plt.ylabel('Log(Wages)')
plt.show()


# More code for regression and further analysis would follow
# Define the independent variables X and the dependent variable y
""" OLS Variables """
"""
X = df[['union', 'exper', 'expersq', 'educ', 'black', 'hisp', 'married', 
        'agric', 'construc', 'manuf', 'nrtheast', 'nrthcen', 'south', 'poorhlth', 'hours']]
"""

"""
X = df[['union', 'exper', 'expersq', 'educ', 'black', 'hisp', 'married', 
        'agric', 'construc', 'manuf', 'nrtheast', 'nrthcen', 'south']]  # Include other relevant variables (if necessary)
y = df['lwage']
"""

""" Robustness Checks """
"""
# Interacting Terms for Robustness Checks
df['union_educ'] = df['union'] * df['educ']

# Define the model with the interaction term
X = df[['union', 'exper', 'expersq', 'educ', 'union_educ', 'married', 'construc', 'manuf', 'nrtheast', 'nrthcen', 'south' ]]  # Add other variables if needed
y = df['lwage']
""" 

""" Fixed Effects Model """
# Ensuring that 'nr' is set as the panel variable and 'year' as the time variable
df = df.set_index(['nr', 'year'])

# Define the model
X = df[['union', 'exper', 'expersq', 'married', 
        'agric', 'construc', 'manuf', 'nrtheast', 'nrthcen', 'south', 'hours']]  # Add other variables if needed
y = df['lwage']

# Run the fixed effects model
# fixed_effects_model = PanelOLS(y, X, entity_effects=True).fit()
""" """

# Add a constant to the model (intercept)
X = sm.add_constant(X) # For OLS

# Create a model and fit it
model = sm.OLS(y, X).fit() #OLS MODEL


""" 2SLS REGRESSION """

# Define your regression variables
exog_vars = ['exper', 'expersq', 'educ', 'black', 'hisp', 'married', 
             'agric', 'construc', 'manuf', 'nrtheast', 'nrthcen', 'south']
exog = sm.add_constant(df[exog_vars])
endog = df['union']
dep = df['lwage']

# Hypothetical Instrumental Variable (replace this with your real IV)
# Assuming 'industry' is a categorical variable with industry codes
# and 'hist_union_rate' is a DataFrame with historical unionization rates by industry
df = df.join(hist_union_rate, on='industry')  
instr = df['industry_unionization_rate']  # historical unionization rate by industry

# Perform the 2SLS regression
iv_model = IV2SLS(dependent=dep, exog=exog, endog=endog, instruments=instr).fit()

# Print out the statistics from the 2SLS regression


""" PLOTTING RESIDUAL """
# Residual Plot fitted regression model
predicted_values = model.fittedvalues
residuals = model.resid

plt.figure(figsize=(8, 5))
plt.scatter(predicted_values, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Log Wages')
plt.ylabel('Residuals')
plt.title('Residual Plot')
# plt.show()



# Print out the statistics
# print(model.summary()) #OLS
# print(fixed_effects_model.summary) # Fixed Effects model
print(iv_model.summary) # 2SLS Model
