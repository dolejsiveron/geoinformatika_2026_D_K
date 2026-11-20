
#KD-tree in Python

#Results
#Processing time for counting density: 3.17  seconds
#Processing time for counting a curvature: 108.23  seconds
#Counted density: 6060.3743

from logging import root
from math import *
from time import *
from matplotlib.pyplot import *
from numpy import * 
from collections import defaultdict
import datetime
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

#Define class for nodes in tree
class KDNode:
    #Defined atributes with know value (if not specified –> none)
    def __init__(self, point=None, axis=None, left=None, right=None, point_indices=None):
        
        #Coordinates X, Y, Z
        self.point = point
        #Which axis divides the points in this node (0=X, 1=Y, 2=Z)
        self.axis = axis
        #Which node is on left to this node
        self.left = left
        #Which node is on right to this node
        self.right = right
        #List of nodes that the node contains
        self.point_indices = point_indices
        
        
class KDTree:

    def __init__(self, data):
        
        self.data = array(data)  # Ensure data is a NumPy array
        
        #Built tree from added data, depth = 0
        self.root = self._build_tree(self.data, 0)
    
    #Divide points by axis and create nodes    
    def _build_tree(self, indices, depth):
        
        #Be sure the dataset contains data, mainly for recursive solution at the end, it breaks the cycle, 
        #AI corrected the code here
        if len(indices) == 0:
            return None 
        
        #Get function that returns 0, 1, 2 and again 0, 1, 2 (see axis in KD note)
        #At the end call function with depth + 1 
        axis_n = depth%3
        
        #Sort coordinate lists numerically by X then Y and then Z (to get median), AI corrected the code here
        indices = sorted(indices, key=lambda p: p[axis_n])
        
        #Median (count//2), get value of the point in the middle
        median_index = len(indices)//2 #It has to be integer, if float, the right side number will be returned
        median_point_value = indices[median_index]
        
        #Function will return all the nodes, AI used here for understanding and correcting code
        return KDNode(
            point = median_point_value,
            axis = axis_n,
            #Recursive, finding left median
            left = self._build_tree(indices[:median_index], depth +1),
            right = self._build_tree(indices[median_index + 1:], depth + 1),
            #Store indentification of the point
            point_indices = median_point_value[3])
        
    #Define function to get points nearest neighbour 
    #root = root of the created KDtree
    #query_point = point for which we are searching the NN
    def getNN (self, root, query_point):
            
        #For now, we dont know the NN
        final_point = None
        final_dist = float('inf')
            
        #Define function for finding it, AI corrected the code here 
        def search (node):
            #Working with var from outer function
            nonlocal final_point, final_dist

            #For stoping recursive action (base case)
            if node is None:
                return 
                
            #Computing the distance
            dx = query_point[0] - node.point[0]
            dy = query_point[1] - node.point[1]
            dz = query_point[2] - node.point[2]
            dist = (dx**2 + dy**2 + dz**2)**(1/2) 
            
            #Update the final distance, not if points are identical     
            if node.point[3] != query_point[3] and dist < final_dist:
                final_dist = dist
                final_point = node.point 
            
            #Get axis
            axis = node.axis
            #Compare 
            difference = query_point[axis] - node.point[axis] 
               
            #Choosing the branch
            if difference < 0:
                near_branch = node.left
                far_branch = node.right
            else:
                near_branch = node.right
                far_branch = node.left

            #Recursive
            search(near_branch)
            
            #Recursive
            #if distance to the dividing line is smaller than distance to NN, we have to search another branch as well
            if abs(difference) < final_dist:
                search(far_branch)
                
        search(root)
        
        return final_point, final_dist 
    
    #Get NN to compute curvature, k means the number of nearest neighbours (30 from assignment)
    #Define function to get points nearest neighbour 
    #root = root of the created KDtree
    #query_point = point for which we are searching the NN
    def getKNN (self, root, query_point, k):
        
        #Create new list for storing distances
        dist_KNN = []
        
        def search (node):
            
            nonlocal dist_KNN, k 
            
            #For stoping recursive action (base case)
            if node is None:
                return
            
            #Calculate distance
            dx = query_point[0] - node.point[0]
            dy = query_point[1] - node.point[1]
            dz = query_point[2] - node.point[2]
            dist = (dx**2 + dy**2 + dz**2)**(1/2)
            
            #Dont include the point itself
            if node.point[3] != query_point[3]:
                #Add K nearest neighbours to the list, AI corrected the code here
                dist_KNN.append((node.point, dist))
                #Sort the list by distance
                dist_KNN.sort(key=lambda x: x[1])
                #If the list is longer than k, remove the farthest point
                if len(dist_KNN) > k:
                    dist_KNN.pop()
            
            #Get axis (0=X, 1=Y, 2=Z)
            axis = node.axis
            #Compare 
            difference = query_point[axis] - node.point[axis] 
                           
            #Choosing the branch
            if difference < 0:
                near_branch = node.left
                far_branch = node.right
            else:
                near_branch = node.right
                far_branch = node.left
            
            search(near_branch)
            
            #Recursive
            #We have to search another branch as well if distance to the dividing line is smaller than distance to NN
            #Or if the list of nearest neighbours is not full yet (less than k), AI corrected the code here
            if len(dist_KNN) < k or abs(difference) < dist_KNN[-1][1]:
                search(far_branch)
                
        #Start the recursive search
        search(root)
        
        return dist_KNN

#For each point, get k nearest neighbours and compute curvature, AI corrected the code here
def compute_curvature(points, knn, tree):
    curvature_list = []
    for i in range(len(points)):
        query_point = points[i]
        #Get k nearest neighbours and distances
        knn_points = tree.getKNN(tree.root, query_point, knn)
        
        #Matrix needs at least 2 points to compute curvature
        if len(knn_points) < 2:
            curvature_list.append(0)
            continue
        
        #Get coordinates of the k nearest neighbours
        knn_coordinates = array([point[0][:3] for point in knn_points])

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

#Starting the computing
print('KD-Tree')
print('Loading points')

#Load points 
X, Y, Z = loadPoints('tree_18.txt')

#Get data format which can be use for KDtree, add index
points = []
for i in range(len(X)):
    #i is index
    points.append([X[i], Y[i], Z[i], i])
    
#Start building the tree with loaded data    
tree = KDTree(points)

#Compute density 
print("Computing density")

start_time_density = process_time()    

def density (points, tree):
    sum_dist = 0
    processed_points = 0
    
    #To compute density we need to get the nearest neighbour for each point
    for i in range(len(points)):
        query_point = points[i]
        #Get nearest neighbour and distance
        nearest_p, distance = tree.getNN(tree.root, query_point)
        
        #Add distance to the sum
        sum_dist = sum_dist + distance
        processed_points = processed_points + 1

    #Check if any points were processed to avoid division by zero
    if processed_points > 0:
        #Compute density
        d_aver = sum_dist / processed_points
        rho = 1 / (d_aver**3)
    else:
        rho = 0
        
    return rho

rho = density(points, tree)
    
end_time_density = process_time()

print("Density computed in: ", end_time_density - start_time_density, " seconds")

#Print information (only density, curvature is not printed because of its long list of values, instead it is visualised)
print("Density: ", rho)

#Starting curvature computing
print("Computing curvature")

#Define the number of NN needed for computation (from assignment)
knn = 30

#Get the start time
start_time_curvature = process_time()

curvature_list = compute_curvature(points, knn, tree)

#Get the end time 
end_time_curvature = process_time()

print("Curvature computed in: ", end_time_curvature - start_time_curvature, " seconds")    

#"""
#Comment here if you dont want to visualise the function and curvature
#Create visualisation of the point cloud with curvature values, AI was used here
print("Generating visualisation of the point cloud with curvature values")
fig = figure()

#Get plot with three axes
ax = axes(projection='3d')

#Color scale is reversed
values = ax.scatter(X, Y, Z, c=curvature_list, cmap='plasma_r', marker='.')

fig.colorbar(values, ax=ax, label='Curvature visualised by color')
show()

#Create plot
print("Generating plot for the relationship between computation time and the number of points")

#Visualise function 
plot_y_time = []
plot_x_numbers = []
plot_x_axis = [100, 2000, 5000, 7000, 10000]

X, Y, Z = loadPoints('tree_18.txt')

#Get data format which can be use for KDtree, add index
points_plot = []
for i in range(len(X)):
    #i is index
    points_plot.append([X[i], Y[i], Z[i], i]) 

#Getting processing time for each value of points in plot_x_axis 
for i in plot_x_axis:  
    
    #Get X axis values for the plot
    plot_x_numbers.append(i)
    
    #Get start time 
    time_start_plot = process_time()
    
    #Built tree for the current number of points
    example_tree = KDTree(points_plot[:i])
    
    #Get density 
    plot_density = density(points_plot[:i], example_tree)
    
    #Get curvature for the current number of points
    knn = 30
    plot_curvature = compute_curvature(points_plot[:i], knn, example_tree)
    
    #Get end time
    time_end_plot = process_time()
    
    plot_y_time.append(time_end_plot - time_start_plot)
    
#Plot visualisation 
print('X axis:', plot_x_numbers)
print('Y axis:', plot_y_time)
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
#Comment here if you dont want to visualise the function and curvature