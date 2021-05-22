#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA)
# 
# Exploratory Data Analysis is a process of examining or understanding the data and extracting insights or main characteristics of the data. EDA is generally classified into two methods, i.e. graphical analysis and non-graphical analysis.
# 
# EDA is very essential because it is a good practice to first understand the problem statement and the various relationships between the data features

# #### Step - 1 - Introduction : Data description and objective
# 
# The dataset was released by Aspiring Minds from the Aspiring Mind Employment Outcome 2015 (AMEO)
# 
# The dataset contains the employment outcomes of engineering graduates as dependent variables (Salary, Job Titles, and Job Locations) along with the standardized scores from three different areas – cognitive skills, technical skills and personality skills
# 
# Today our objective is to process the EDA for the given data and get to know about various relationships between the data.

# #### Step - 2 - Import the data and display the head, shape and description of the data.

# In[48]:


#importing python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#importing data
df = pd.read_excel(r'C:\Users\dell\Desktop\Innomatics\Task-10 (Exploratory Data Analysis)\aspiring_minds_employability_outcomes_2015.xlsx')
df.head()


# ##### head() is used for displaying the top 5 rows of the dataframe

# In[49]:


df.shape


# ##### .shape is used for Displaying the shape of the data, it returns the values of number of rows,coloumns

# In[50]:


df.tail()


# ##### tail() is used for displaying the last 5 rows of the dataframe

# In[51]:


df.describe()


# ##### describe() returns the values of mean,std,min and max on all the coloumns

# In[52]:


df.info()


# ##### info() returns the total data description(coloumn name, non null, count and datatype) of the dataframe and gives a total overview of the dataframe

# #### Step - 3 - Univariate Analysis -&gt; PDF, Histograms, Boxplots, Countplots, etc..
# - Find the outliers in each numerical column
# - Understand the probability and frequency distribution of each numerical column
# - Understand the frequency distribution of each categorical Variable/Column

# In[53]:


df.isnull().sum()


# ###### We can clearly see that we don't have any missing values at all so we happily do explore the data in various ways

# In[54]:


df.dtypes


# #### Step-3 Univariate Analysis
# 
# Univariate analysis is the simplest form of analyzing data. “Uni” means “one”, so in other words your data has only one variable.
# 
# Tables, charts, polygons, and histograms are all popular methods for displaying univariate analysis of a specific variable.
# 
# Now lets check histograms for our data

# In[55]:


plt.figure(figsize=(10,5))
df['Salary'].hist()


# In[56]:


plt.figure(figsize=(10,5))
df['Gender'].hist()


# In[57]:


plt.figure(figsize=(10,5))
df['Degree'].hist()


# In[58]:


plt.figure(figsize=(10,5))
df['Specialization'].hist()


# Observations from above histograms
# 1. we can see that among all engineering graduates most of people who graduated from colleges are Male than female.
# 2. we can see job salary in any field is an average of 2.5LPA
# 3. we can see among all degrees, there are more students from btech/be than other degrees

# In[59]:


#Analysis of personality test using subplot
plt.figure(figsize=(15,10))
plt.subplot(221)
sns.distplot(df['agreeableness'])
plt.title("Normal Plot of agreeableness Score")
plt.subplot(222)
sns.distplot(df['extraversion'])
plt.title("Normal plot of extraversied Score")
plt.subplot(223)
sns.distplot(df['nueroticism'])
plt.title("Normal plot of neurotics score")
plt.subplot(224)
sns.distplot(df['openess_to_experience'])
plt.title("Normal plot of Experience score")
plt.show()


# ###### Using subplot we can plot different plots in one single plot.

# ###### Lets find Outliers

# In[60]:


plt.figure(figsize=(19,12))


num_features = df.select_dtypes(include=['int64']).columns

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.boxplot(df[num_features[i]])
    plt.title(num_features[i],color="b",fontsize=20)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)


# In[61]:


lower_bound =0.1
upper_bound=0.95
res = df.quantile([lower_bound,upper_bound])
res


# In[62]:


from scipy.stats.mstats import winsorize
df["ID"]           = winsorize(df["ID"],(0,0.10))
df["Salary"]        = winsorize(df["Salary"],(0,0.10))
df["12graduation"]  = winsorize(df["12graduation"],(0,0.099))
df["CollegeID"]  = winsorize(df["CollegeID"],(0,0.099))
df["CollegeTier"]  = winsorize(df["CollegeTier"],(0,0.099))
df["CollegeCityID"]= winsorize(df["CollegeCityID"],(0.10,0.20))


# In[63]:


plt.figure(figsize=(19,12))


num_features = df.select_dtypes(include=['int64']).columns

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.boxplot(df[num_features[i]])
    plt.title(num_features[i],color="b",fontsize=20)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)


# ###### After performing winsoring method we can clearly see the reduction of outliers

# In[64]:


job_c = df['JobCity'].value_counts()
job_c.plot(kind="line",figsize=(20,10))


# ##### from this line plot, we can clearly see that more number of software employees are likely to work at Bangalore as jobcity than other cities

# #### Step-4 Bivariate Analysis
# 
# Bivariate analysis is one of the simplest forms of quantitative (statistical) analysis. 
# 
# It involves the analysis of two variables (often denoted as X, Y), for the purpose of determining the empirical relationship between them

# In[65]:


# Heat Map

#Correlation between variables
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.show()


# In[66]:


#Degree and gender
sns.jointplot(y=df['Degree'],x=df['Gender'],data=df)


# In[67]:


sns.jointplot(x=df['Degree'],y=df['collegeGPA'],data=df,kind="scatter")


# In[68]:


sns.jointplot(x=df['Salary'],y=df['collegeGPA'],data=df,kind="hex")


# In[69]:


sns.jointplot(x=df['12percentage'],y=df['10percentage'],kind='hex',data=df)


# ###### observations from bivariate analysis
# 1. heat map explains the correlation of data
# 2. male employees are more than female employees
# 3. salary range and CGPA is equally spread, that means salary is not dependant on CGPA
# 4. candiadates has scored more than 60 percent in bot 10 and 12 standards

# #### Step-5 Research question:
# 
# Times of India article dated Jan 18, 2019 states that “After doing your Computer Science Engineering if you take up jobs as a Programming Analyst, Software Engineer, Hardware Engineer and Associate Engineer you can earn up to 2.5-3 lakhs as a fresh graduate.” Test this claim with the data given to you.Is there a relationship between gender and specialisation? (i.e. Does the preference of Specialisation depend on the Gender?)
# 
# Let us make a bold claim that gender and specialisation are dependent $\\
#    \\ Alternate\ hypothesis:$$$\ H1 = They\ are\ dependent$$Null hypothesis:$$\ H0= They\ are\ independent$$
#    
# Let us use hypothesis testing for this question find out the claim

# In[70]:


pd.crosstab(df['Specialization'],df['Gender'],margins = True)


# In[71]:


observed=pd.crosstab(df['Specialization'],df['Gender'])


# In[72]:


from scipy.stats import chi2
from scipy.stats import chi2_contingency
chi2_contingency(observed)


# In[73]:


chi2_test_statistic = chi2_contingency(observed)[0]
pval = chi2_contingency(observed)[1]
df1 = chi2_contingency(observed)[2]


# In[74]:


#Calculating chi critical
confidence_level =0.90
alpha = 1 - confidence_level
chi2_critical = chi2.ppf(1-alpha,df1)
chi2_critical


# In[75]:


#Plotting the chi2 distribution
x_min = 0
x_max = 100

#plotting the graph and setting the limits
x = np.linspace(x_min,x_max,100)
y = chi2.pdf(x,df1)
plt.xlim(x_min,x_max)
plt.plot(x,y)

#Setting chi critical value
chi_critical_right = chi2_critical

#Shading the rejection region
x1 = np.linspace(chi_critical_right,x_max,100)
y1 = chi2.pdf(x1,df1)
plt.fill_between(x1,y1,color='red')


# In[76]:


#conclusion with chi2 test
if(chi2_test_statistic>chi2_critical):
    print("Reject null hypothesis")
else:
    print("Failed to reject null hypothesis")


# In[77]:


#Conclusion with p-test
if (pval<alpha):
    print("Reject null hypothesis")
else:
    print("Failed to reject null hypothesis")


# ## Conclusion
# 
# ##### From this assignment i have drewn a couple of conclusions noted below
# 
# 1. Using exploratory data analysis, we have analyzed and investigated the dataset and summarized few characteristics of the data.
# 2. We have used data vusualization methods, univariate and bivariate methods for visualization the data.
# 3. These data visualization methods helped us to determine how best to manipulate data sources to get the answers you need, making it easier to discover patterns, test a hypothesis, or check assumptions.
# 4. We have used Heat map, which is a graphical representation of data where values are depicted by color.
# 5. Also we have used Scatter plot, which is used to plot data points on a horizontal and a vertical axis to show how much one variable is affected by another.
# 6. Apart from the above conclusions, I have mentioned observations about the dataset and mentioned the same at each and every plot and also at the end of univariate and bivariate analysis.

# #### THANK YOU

# In[ ]:




