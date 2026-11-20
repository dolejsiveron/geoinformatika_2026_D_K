
#Naive method in Python

#Results
#Processing time for counting density: 102.78 seconds
#Counted density: 5881.4472

#Import libraries 
from math import *
from time import *
from matplotlib.pyplot import *
from numpy import * 

#Load points from file to correct format (3 lists) 
def loadPoints(file):
    #Load file
    X, Y, Z = [], [], [] ;
    
    with open(file) as f:
        
        for line in f:
            x, y, z = line.split('\t')
            
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))
    
    return X, Y, Z

#Find nearest point and its distance (xq, yq, zq are the coordinates of the one specific point
#X, Y, Z are lists of all coordinates, 
#return: coordinates of NN and minimum distance
def getNN(xq, yq, zq, X, Y, Z):
    
    dmin = inf
    xn, yn, zn = X[0], Y[0], Z[0]
    
    for i in range(len(X)):
        #Compute distance
        dx, dy, dz = xq - X[i], yq - Y[i], zq - Z[i]
        d = (dx*dx + dy*dy + dz * dz)**0.5
        
        #Actualize minimum: distance + coordinates
        if d < dmin and d > 0:
            dmin = d
            xn, yn, zn = X[i], Y[i], Z[i]
    return xn, yn, zn, dmin

#Start the computing and load points from PC
print('Naive method')
print("Loading points from file")
X, Y, Z = loadPoints('tree_18.txt')

#Start of the implementation
print('Computing density of the point cloud')

#Get the start time 
start_time = process_time()

#Get average distance to nearest point
def density (X, Y, Z):
    d_sum = 0
    n = len(X)
    for i in range(n):
        xn, yn, zn, dmin = getNN(X[i], Y[i], Z[i], X, Y, Z)
        
        #Sum of all distances 
        d_sum = d_sum + dmin   
        
    #Get average distance
    d_aver = d_sum / n

    #Get density (do not divide by zero)
    if d_aver > 0:
        ro_density = 1/(d_aver**3)
    else:
        ro_density = 0
    return  ro_density

#Get density
ro_density = density(X, Y, Z)

#Get elapsed time     
end_time = process_time()


#Get all the information about computing 
print('Density:',ro_density)
print('Elapsed:', end_time - start_time, 'seconds')

#"""
#Comment here if you dont want to visualise the function
#Visualise function 
plot_y_time = []
plot_x_numbers = []
plot_x_axis = [100, 2000, 5000, 7000, 10000]

#Getting processing time for each value of points in plot_x_axis 
print('Generating plot for the relationship between computation time and the number of points')

for i in plot_x_axis:
    X_plot = X[:i]
    Y_plot = Y[:i]
    Z_plot = Z[:i]
    
    plot_x_numbers.append(i)
    
    #Get start time 
    time_start_plot = process_time()
    
    #Compute density (doesnt need to be saved)
    density(X_plot, Y_plot, Z_plot)
    
    #Get end time
    time_end_plot = process_time()
    
    plot_y_time.append(time_end_plot - time_start_plot)
    
#Print values used for plot
print('X axis:', plot_x_numbers)
print('Y axis:', plot_y_time)

#Visualise the plot 
x = array(plot_x_numbers)
y = array(plot_y_time)  
plot(x, y, linestyle = 'dotted')
xlabel('Number of points') 
ylabel('Processing time (s)')
title('Graph of the relationship between computation time and the number of points')
grid(True)
    
scatter(x, y)
show()

#"""