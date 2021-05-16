#!/usr/bin/env python
# coding: utf-8

# #### Objective 1
# ###### In this challenge, we learn about binomial distributions. 
# 
# #### Task
# ###### The ratio of boys to girls for babies born in Russia is 1.09:1 . If there is 1 child born per birth, what proportion of Russian families with exactly 6 children will have at least 3 boys?
# 
# ###### Write a program to compute the answer using the above parameters. Then print your result, rounded to a scale of  decimal places (i.e.,  1.234 format).

# In[11]:


boys,girls = [float(x) for x in input().strip().split(' ')]

probGirls = (girls/(boys + girls))**4 *(boys/(boys + girls)) ** 2 *15 + (girls/(boys + girls))**5 * (boys/(boys+girls)) *6 + (girls/(boys+girls))**6
probBoys = 1 - probGirls
print("%.3f" %probBoys)


# #### Objective 2
# ###### In this challenge, we go further with binomial distributions.
# 
# #### Task
# ###### A manufacturer of metal pistons finds that, on average, 12% of the pistons they manufacture are rejected because they are incorrectly sized. What is the probability that a batch of 10 pistons will contain:
# 
# ###### No more than 2 rejects?
# ###### At least 2 rejects?

# In[3]:


from math import factorial

p,n = [float(x) for x in input().strip().split(' ')]

p = p/100.0
q = 1-p

B0 = factorial(n)/(factorial(0)*factorial(n-0)) * p**0 * q**(n-0)
B1 = factorial(n)/(factorial(1)*factorial(n-1)) * p**1 * q**(n-1)
B2 = factorial(n)/(factorial(2)*factorial(n-2)) * p**2 * q**(n-2)

print(round(B0+B1+B2,3))
print(round(1-(B0+B1),3))


# #### Objective 3
# ###### In this challenge, we learn about normal distributions.
# 
# #### Task
# ###### In a certain plant, the time taken to assemble a car is a random variable, X , having a normal distribution with a mean of 20 hours and a standard deviation of 2 hours. What is the probability that a car can be assembled at this plant in:
# 
# ###### Less than 19.5 hours?
# ###### Between 20 and 22 hours?

# In[5]:


import math
mean,std = [int(x) for x in input().strip().split(' ')]
less_hour=float(input())
between1,between2 = [int(x) for x in input().strip().split(' ')]

result1 = 1/2 * (1 + math.erf((less_hour - mean)/(math.sqrt(std)*2)))
result2 = (1/2 * (1 + math.erf((between2 - between1)/(math.sqrt(std)*2)))) - 1/2
print("%.3f" % result1)
print("%.3f" % result2)


# #### Objective 4
# ###### In this challenge, we go further with normal distributions. 
# 
# #### Task
# ###### The final grades for a Physics exam taken by a large group of students have a mean of 70 and a standard deviation of 10 . If we can approximate the distribution of these grades by a normal distribution, what percentage of the students:
# 
# ###### Scored higher than 80 (i.e., have a grade>80)?
# ###### Passed the test (i.e., have a grade>=60)?
# ###### Failed the test (i.e., have a grade<60)?
# ###### Find and print the answer to each question on a new line, rounded to a scale of 2 decimal places.

# In[7]:


import math
mean,std = [int(x) for x in input().strip().split(' ')]
high_mark=int(input())
threshold_mark= int(input())

grade_more_than_80 = 1 - (.5 * (1 + math.erf((high_mark - mean)/(math.sqrt(std)*10))))
pass_grade = 1 - (.5 * (1 + math.erf((threshold_mark - mean)/(math.sqrt(std)*10))))
fail_grade = .5 * (1 + math.erf((threshold_mark - mean)/(math.sqrt(std)*10)))

print("%.2f" %(grade_more_than_80*100))
print("%.2f" %(pass_grade*100))
print("%.2f" %(fail_grade*100))


# #### Objective 5
# ###### In this challenge, we practice solving problems based on the Central Limit Theorem. 
# 
# #### Task
# ###### A large elevator can transport a maximum of 9800 pounds. Suppose a load of cargo containing 49 boxes must be transported via the elevator. The box weight of this type of cargo follows a distribution with a mean of 205 pounds and a standard deviation of 15 pounds. Based on this information, what is the probability that all 49 boxes can be safely loaded into the freight elevator and transported?

# In[9]:


import math

max_weight = 9800
n = 49
mean = 205
std = 15
result= 0.5 * (1 + math.erf((max_weight - (n*mean))/((math.sqrt(n) * std) * math.sqrt(2))))
print("%.4f" % round(result, 4))


# #### Objective 6
# ###### In this challenge, we practice solving problems based on the Central Limit Theorem. 
# 
# #### Task
# ###### The number of tickets purchased by each student for the University X vs. University Y football game follows a distribution that has a mean of 2.4 and a standard deviation of 2.
# 
# ###### A few hours before the game starts, 100 eager students line up to purchase last-minute tickets. If there are only 250 tickets left, what is the probability that all 100 students will be able to purchase tickets?

# In[10]:


import math

student_count = 250
n = 100
mean = 2.4
std = 2.0
result=0.5*(1+(math.erf((student_count-(n*mean))/((math.sqrt(n) * std)*math.sqrt(2)))))
print("%.4f" % round(result,4))


# #### Objective 7
# ###### In this challenge, we practice solving problems based on the Central Limit Theorem. 
# 
# #### Task
# ###### You have a sample of 100 values from a population with mean 500 and with standard deviation 80. Compute the interval that covers the middle 95% of the distribution of the sample mean; in other words, compute A and B such that P(A<x<B)=0.95. Use the value of z=1.96. Note that z is the z-score.

# In[11]:


import math
samples = 100
mean = 500
std = 80
std = std / math.sqrt(samples)
interval = .95
z = 1.96
p_of_a = -z * std + mean
p_of_b = z * std + mean
print("%.2f" % round(p_of_a,2))
print("%.2f" % round(p_of_b,2))


# #### Objective 8
# ###### In this challenge, we practice calculating the Pearson correlation coefficient. 
# 
# #### Task
# ###### Given two n-element data sets, X and Y, calculate the value of the Pearson correlation coefficient.

# In[13]:


# Enter your code here. Read input from STDIN. Print output to STDOUT

n = int(input().strip())
X = [float(x) for x in input().strip().split(' ')]
Y = [float(x) for x in input().strip().split(' ')]
mean_of_X = sum(X)/n
mean_of_Y = sum(Y)/n
variance_of_X = 0
for x in X:
    variance_of_X = variance_of_X + (x - mean_of_X)**2
variance_of_X = variance_of_X * 1/n
std_of_X = variance_of_X**.5
variance_of_Y = 0
for y in Y:
    variance_of_Y = variance_of_Y + (y - mean_of_Y)**2
variance_of_Y = 1/n * variance_of_Y
std_of_Y = variance_of_Y**.5
summation = 0
for i in range(n):
    summation = summation + (X[i] - mean_of_X)*(Y[i] - mean_of_Y)
rho_X_Y = summation/(n*std_of_X *std_of_Y)
print("%.3f" % rho_X_Y)


# #### Objective 9
# ###### In this challenge, we practice using linear regression techniques. 
# 
# #### Task
# ######  A group of five students enrolls in Statistics immediately after taking a Math aptitude test. Each student's Math aptitude test score, x, and Statistics course grade, y, can be expressed as the following list of (x,y) points:
# 1. (95,85)
# 2. (85,95)
# 3. (80,70)
# 4. (70,65)
# 5. (60,70)
# ###### If a student scored an 80 on the Math aptitude test, what grade would we expect them to achieve in Statistics? Determine the equation of the best-fit line using the least squares method, then compute and print the value of y when x=80.

# In[14]:


n = 5
X = [95,85,80,70,60]
Y = [85,95,70,65,70] 
mean_of_X = sum(X)/n
mean_of_Y = sum(Y)/n
variance_of_X = 0
for x in X:
    variance_of_X = variance_of_X + (x - mean_of_X)**2
variance_of_X = variance_of_X * 1/n
std_of_X = variance_of_X**.5
variance_of_Y = 0
for y in Y:
    variance_of_Y = variance_of_Y + (y - mean_of_Y)**2
variance_of_Y = 1/n * variance_of_Y
std_of_Y = variance_of_Y**.5
summation = 0
for i in range(n):
    summation = summation + (X[i] - mean_of_X)*(Y[i] - mean_of_Y)
rho_X_Y = summation/(n*std_of_X *std_of_Y)

y = rho_X_Y * std_of_Y / std_of_X
x = mean_of_Y - y * mean_of_X
print("%.3f" % (x + y*80))


# #### Objective 10
# ###### In this challenge, we practice using multiple linear regression

# In[1]:


from sklearn import linear_model

m,n = [int(x) for x in input().split(' ')]
x = []
y = []
for i in range(n):
    arr = [float(x) for x in input().split(' ')]
    x.append(arr[:-1])
    y.append(arr[-1])

lm = linear_model.LinearRegression()
lm.fit(x, y)
a = lm.intercept_
b = lm.coef_
q = int(input())
for i in range(q):
    arr = [float(x) for x in input().split(' ')]
    ans = a
    for j in range(m):
        ans = ans + arr[j]*b[j]
    print("%.2f" % ans)


# In[ ]:




