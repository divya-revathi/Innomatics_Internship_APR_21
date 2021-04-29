# -*- coding: utf-8 -*-
"""Task 1(Python Programming).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qltR9ChVs8n2Aa-hwm31oPG5TxLWmFne
"""

#Ques-1: Print Hello, World! to stdout.

print("Hello, World!")

#Ques-2: Given an integer,n, perform the following conditional actions:
#If n is odd, print Weird
#If n is even and in the inclusive range of 2 to 5, print Not Weird
#If n is even and in the inclusive range of 6 to 20, print Weird
#If n is even and greater than , print Not Weird


import math
import os
import random
import re
import sys


n= int(input())
if(n>0 and n<=100):
 if(n%2==0):
  if(n>=6 and n<=20):
   print("Weird")
  else: print("Not Weird")
 else: print("Weird")
else: exit(0)

#Ques-3: The provided code stub reads two integers from STDIN, a and b. Add code to print three lines where:
#The first line contains the sum of the two numbers.
#The second line contains the difference of the two numbers (first - second).
#The third line contains the product of the two numbers.


a = int(input())
b = int(input())
print(a+b)
print(a-b)
print(a*b)

#Ques-4: The provided code stub reads two integers, a and b, from STDIN.
#Add logic to print two lines. The first line should contain the result of integer division, a//b . The second line should contain the result of float division,  a/b .
#No rounding or formatting is necessary.


a = int(input())
b = int(input())

print(a//b)
print(a/b)

#Ques-5: For all non-negative integers i<n, print the square of each number on a separate line.


n=int(input())
if(n>0 and n<=20):
 for x in range(n):
  print(x*x)
  x+=1
else: exit(0)

#Ques-6: Given a year, determine whether it is a leap year. If it is a leap year, return the Boolean True, otherwise return False.
#Note that the code stub provided reads from STDIN and passes arguments to the is_leap function. It is only necessary to complete the is_leap function.


def is_leap(year):
    leap = False
    
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                leap=True
            else:
                leap=False
        else:
             leap=True
    else:
        leap=False


    
    return leap

year = int(input())
print(is_leap(year))

#Ques-7: Print the list of integers from 1 through n as a string, without spaces.


n = int(input())

for x in range(n):
    print(x+1 , end="")