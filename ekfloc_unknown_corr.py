# -*- coding: utf-8 -*-
"""EKFLOC_KNOWN_CORR.ipynb

Created by Priyanka Das
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import csv
cols = 4
rows = 2113
Range1=[]
Range2=[]
Range3=[]
Range4=[]
Bearing1=[]
Bearing2=[]
Bearing3=[]
Bearing4=[]
X_True=[]
Y_True=[]
Theta_True=[]
Uk=[]
Wk=[]

with open("unknown_correspondence_sensor_landmarks.csv","r") as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  count=0
  Range1=[]
  for lines in csv_reader:
    if count==0:
      count=1
    else:
      Range1.append(float(lines[1]))
      Range2.append(float(lines[3]))
      Range3.append(float(lines[5]))
      Range4.append(float(lines[7]))
      Bearing1.append(float(lines[2]))
      Bearing2.append(float(lines[4]))
      Bearing3.append(float(lines[6]))
      Bearing4.append(float(lines[8]))
Range=[[Range1],[Range2],[Range3],[Range4]]
Bearing=[[Bearing1],[Bearing2],[Bearing3],[Bearing4]]

     
with open("unknown_correspondence_true_odometry.csv","r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            X_True.append(float(row[1]))
            Y_True.append(float(row[2]))
            Theta_True.append(float(row[3]))
            Uk.append(float(row[4]))
            Wk.append(float(row[5]))
            line_count += 1
T=0.033
State=np.array([[-2],[-0.5],[1.977]])
Qt=np.array([[1.5, 0],[0, 1.5]])
P=np.identity(3)
Mx=np.array([-2.5, -2.5, 2.5, 2.5]) #Coordinates for landmarks
My=np.array([2.5, -1.0, -1.0, 2.5])

X_est=[]
Y_est=[]
Theta_est=[]
Delta=np.zeros(2)
a1=0.3 #tuning parameters
a2=0.5
a3=0.3
a4=0.5

def EKF(State,Qt,P,Mx,My,Uk, Wk,Range, Bearing,T,i):
  R=Uk[i]/Wk[i]
  State_upt=State+[[-R*math.sin(State[2])+R*math.sin(State[2]+Wk[i]*T)],[R*math.cos(State[2])-R*math.cos(State[2]+Wk[i]*T)],[Wk[i]*T]]
  Vt=np.array([[(-math.sin(State[2])+math.sin(State[2]+Wk[i]*T))/Wk[i], Uk[i]*(math.sin(State[2])-math.sin(State[2]+Wk[i]*T))/Wk[i]**2+(Uk[i]*math.cos(State[2]+Wk[i]*T)*T/Wk[i])], [(math.cos(State[2])-math.cos(State[2]+Wk[i]*T))/Wk[i], (-Uk[i]*(math.cos(State[2])-math.cos(State[2]+Wk[i]*T))/Wk[i]**2)+(Uk[i]*math.sin(State[2]+Wk[i]*T)*T/Wk[i])], [0, T]])
  G=np.array([[1, 0, -R*math.cos(State[2])+R*math.cos(State[2]+Wk[i]*T)],[0, 1, -R*math.sin(State[2])+R*math.sin(State[2]+Wk[i]*T)],[0, 0, 1]])
  Mt=np.array([[a1*Uk[i]**2+a2*Wk[i]**2, 0],[0, a3*Uk[i]**2+a4*Wk[i]**2]])       
  P_new=G@P@G.transpose()+(Vt@Mt@Vt.transpose())
  State_upt=State_upt.astype(np.float)
  K1=np.array([[0],[0],[0]])
  K2=np.array([[0],[0],[0]])
  for j in range(4):
    Z=np.array([[Range[j][0][i]],[Bearing[j][0][i]]])
    Z=Z.astype(np.float)
    a=math.isnan(Range[j][0][i])
    if a==False:
      J=np.zeros(4)
      for m in range(4):
        Delta[0]=Mx[m]-State_upt[0]
        Delta[1]=My[m]-State_upt[1]
        d=(Delta[0])
        b=(Delta[1])
        q=Delta[0]**2+Delta[1]**2
        Z_hat=np.array([[q**(1/2)],[float(math.atan2(Delta[1],Delta[0]))-State_upt[2]]])
        Z_hat=Z_hat.astype(np.float)
        H=np.array([[-d/math.sqrt(q), -b/math.sqrt(q), 0],[b/q, -d/q, -1]])         
        S_k=H@P_new@H.transpose()+Qt
        J[m]=((np.linalg.det(2*3.14159*S_k))**-0.5) * math.exp(-0.5*(Z - Z_hat).transpose()@np.linalg.inv(S_k)@(Z - Z_hat));
      j=np.argmax(J)

      Delta[0]=Mx[j]-State_upt[0]
      Delta[1]=My[j]-State_upt[1]
      d=(Delta[0])
      b=(Delta[1])
      q=Delta[0]**2+Delta[1]**2
      Z_hat=np.array([[q**(1/2)],[float(math.atan2(Delta[1],Delta[0]))-State_upt[2]]])
      Z_hat=Z_hat.astype(np.float)
      H=np.array([[-d/math.sqrt(q), -b/math.sqrt(q), 0],[b/q, -d/q, -1]])         
      S_k=H@P_new@H.transpose()+Qt
      
      K=P_new@H.transpose()@np.linalg.inv(S_k)
      State_upt=State_upt+K@(Z-Z_hat)
      P_new=(np.identity(3)-K@H)@P_new
  P=P_new
  State=State_upt
  State=State.astype(np.float)
  return (State, P)  


for i in range(2113):
  (State, P)=EKF(State,Qt,P,Mx,My,Uk, Wk,Range, Bearing,T,i)
  X_est.append(State[0])
  Y_est.append(State[1])
  Theta_est.append(State[2])

Time=np.linspace(start = 0, stop = 70.4, num = 2113)
plt.plot(Time,X_True, label='True Position')
plt.plot(Time,X_est, label='Estimated Position')
plt.ylabel('X Position Measurement') 
#naming the y axis 
plt.xlabel('Time') 
  
# giving a title to my graph 
plt.title('X - Position Comparrision') 
plt.legend()
plt.figure(figsize=(30, 30))  
# function to show the plot 
plt.show() 


plt.plot(Time,Y_True, label='True Position')
plt.plot(Time,Y_est, label='Estimated Position')
plt.ylabel('Y Position Measurement') 
#naming the y axis 
plt.xlabel('Time') 
  
# giving a title to my graph 
plt.title('Y - Position Comparrision') 
plt.legend()
plt.figure(figsize=(30, 30))  
# function to show the plot 
plt.show()

plt.plot(Time,Theta_True, label='True Position')
plt.plot(Time,Theta_est, label='Estimated Position')
plt.ylabel('Theta Position Measurement') 
#naming the y axis 
plt.xlabel('Time') 
  
# giving a title to my graph 
plt.title('Theta - Position Comparrision') 
plt.legend()
plt.figure(figsize=(30, 30))  
# function to show the plot 
plt.show()

plt.plot(X_est,Y_est, label='Estimated Position')
plt.plot(X_True,Y_True, label='True Position')
plt.ylabel('Y Position Measurement') 
#naming the y axis 
plt.xlabel('X') 
  
# giving a title to my graph 
plt.title('Actual - Position Comparrision') 
plt.legend()
plt.figure(figsize=(30, 30))  
# function to show the plot 
plt.show()

for y in range (4):
  print(y)
