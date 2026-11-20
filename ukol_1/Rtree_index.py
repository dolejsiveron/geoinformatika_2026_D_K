
#Rtree in Python

#Results
#Processing time for counting density: 1.06 seconds
#Processing time for counting a curvature: 4.64 seconds
#Counted density: 6060.3743

#Importing libraries 
from math import *
from time import *
from matplotlib.pyplot import *
from numpy import * 

#Used library for Rtree implementation
from rtree import *
from rtree.index import Property
from numpy.linalg import *

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
    
#Index points using Rtree
def Rtree(points):
    
    #Define that we are working in 3D space
    p = Property()
    p.dimension = 3
    idx = Index(properties=p)
    
    #Get Rtree index of points
    for i in range(len(points)):
        x, y, z = points[i]
        #Left, bottom, right, top, front, back
        idx.insert(i, (float(x), float(y), float(z), float(x), float(y), float(z))) 
        
    return idx

#Find nearest neighbors of a given point
def getNN (index, point, numNN, point_id):
    x, y, z = point
    #numNN + 1 because the first point is the point itself, AI corrected the code here
    NN = list(index.nearest((float(x), float(y), float(z), float(x), float(y), float(z)), numNN + 1))
    
    # Remove the query point itself, AI suggestion
    NN = [p for p in NN if p != point_id]
    
    return NN[:numNN]  

#Get density of all points in a given area
def Density(idx, points):
    sum_distNN = 0
    for i in range(len(points)):
        nearest = getNN(idx, points[i], 1, i)
        
        #Get coordinates of nearest point 
        nearest_crd = points[nearest[0]]
        
        #Calculate distance between the point and its nearest neighbor
        dx = nearest_crd[0] - points[i][0]
        dy = nearest_crd[1] - points[i][1]
        dz = nearest_crd[2] - points[i][2]
        dist = (dx**2 + dy**2 + dz**2)**(1/2)
        sum_distNN = sum_distNN + dist
    
    #Get density   
    if sum_distNN > 0:
            #Compute density
            d_aver = sum_distNN / len(points)
            rho = 1 / (d_aver**3)
    else:
        rho = 0
        
    return rho

def curvature(index, points, knn):
    curvature_list = []
    for i in range(len(points)):
        query_point = points[i]
        #Get k nearest neighbours and distances
        knn_points = getNN(index, query_point, knn, i)
            
        #Matrix needs at least 3 points to compute curvature
        if len(knn_points) < 3:
            curvature_list.append(0)
            continue
            
        #Get coordinates of the k nearest neighbours
        knn_coordinates = array([points[point][:3] for point in knn_points])
    
        #Compute covariance matrix of the k nearest neighbours
        cov_matrix = cov(array(knn_coordinates).T)
            
        #Compute eigenvalues of the covariance matrix
        eigenvalues= eigvalsh(cov_matrix)
        
        #Compute curvature using the formula: c = λ1 / (λ1 + λ2 + λ3), do not divide by zero
        if sum(eigenvalues) != 0:
            curvature = eigenvalues[0] / sum(eigenvalues)
        else:
            curvature = 0
            
        curvature_list.append(curvature)
            
    return curvature_list

#Start the computing
print("Rtree")    

#Loading points from file
print("Loading points from file")
X, Y, Z = loadPoints("tree_18.txt")

#Get the right format for indexing the data in Rtree 
pointsXYZ =[]
for i in range(len(X)):
    pointsXYZ.append([X[i], Y[i], Z[i]])
    
#Index the data
Rtree_index = Rtree(pointsXYZ)

#Getting density of the points in the area
print("Starting density computation")
start_time_density = process_time()

rho = Density(Rtree_index, pointsXYZ)

end_time_density = process_time()

print("Density computation time: ", end_time_density - start_time_density, "seconds")
print("Density: ", rho)

#Starting curvature computation
print("Starting curvature computation")
start_time_curvature = process_time()

curvature_list = curvature(Rtree_index, pointsXYZ, 30)

end_time_curvature = process_time()
print("Curvature computation time: ", end_time_curvature - start_time_curvature, "seconds")

#"""
#Comment here if you dont want to visualise the function and curvature
#Create visualisation of the point cloud with curvature values, AI was used here
print("Generating visualisation of the point cloud with curvature values")
fig = figure()

#Get plot with three axes
ax = axes(projection="3d")

#Color scale is reversed
values = ax.scatter(X, Y, Z, c=curvature_list, cmap="plasma_r", marker=".")

fig.colorbar(values, ax=ax, label="Curvature visualised by color")
show()

#Create plot
print("Generating plot for the relationship between computation time and the number of points")

#Visualise function 
plot_y_time = []
plot_x_numbers = []
plot_x_axis = [100, 2000, 5000, 7000, 10000]

#Getting processing time for each value of points in plot_x_axis 
for i in plot_x_axis:  
    
    #Get X axis values for the plot
    plot_x_numbers.append(i)
    
    #Get start time 
    time_start_plot = process_time()
    
    #Built tree for the current number of points
    example_tree = Rtree(pointsXYZ[:i])
    
    #Get density 
    plot_density = Density(example_tree, pointsXYZ[:i])
    
    #Get curvature for the current number of points, AI corrected the code here
    knn = 30
    plot_curvature = curvature(example_tree, pointsXYZ[:i], knn)
    
    #Get end time
    time_end_plot = process_time()
    
    plot_y_time.append(time_end_plot - time_start_plot)
    
#Print values used in plot
print("X axis:", plot_x_numbers)
print("Y axis:", plot_y_time)

#Plot visualisation 
x = array(plot_x_numbers)
y = array(plot_y_time)  
plot(x, y, linestyle = "dotted")
xlabel("Number of points") 
ylabel("Processing time (s)")
title("Graph of the relationship between computation time and the number of points")
grid(True)
    
scatter(x, y)
show()

#"""
#Comment here if you dont want to visualise the function and curvature