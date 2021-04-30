# -*- coding: utf-8 -*-
"""Task-3 (Python Programming).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1R6PX4fM6bY0HV9gF5cf5l78vrDYbUW0d
"""

#Ques-1 Polar Cordinates

import cmath
number=complex(input())
print(abs(number))
print(cmath.phase(number))

#Ques-2 MBC Angle

import math

ab = int(input())
bc = int(input())
degree_sign = u"\N{DEGREE SIGN}"
hyp = math.hypot(ab,bc)
angle=math.degrees(math.acos(bc/hyp))
print(str(int(round(angle,0)))+degree_sign)

#Ques-3 Palindrome triangle

for n in range(1,int(input())+1): #More than 2 lines will result in 0 score. Do not leave a blank line also
    print([0, 1, 121, 12321, 1234321, 123454321, 12345654321, 1234567654321, 123456787654321, 12345678987654321][n])

#Ques-4 Divmod function

n1=int(input())
n2=int(input())
print(int(n1/n2))
print(n1%n2)
print(divmod(n1,n2))1

#Ques-5 Power and Mod-Power

a=int(input())
b=int(input())
m=int(input())
print(pow(a,b))
print(pow(a,b,m))

#Ques-6 Print the result of a^b + c^d on one line.

a=int(input())
b=int(input())
c=int(input())
d=int(input())
print(pow(a,b)+pow(c,d))

#Ques-7 Triangle quest

for n in range(1,int(input())): #More than 2 lines will result in 0 score. Do not leave a blank line also
    print([0, 1, 22, 333, 4444, 55555, 666666, 7777777, 88888888, 999999999][n])