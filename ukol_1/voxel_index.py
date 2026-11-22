#Voxel in Python

#Results
#Processing time for counting density: 3.14  seconds
#Processing time for counting a curvature: 13.08 seconds
#Counted density: 6060.3743

from math import *
from time import *
from matplotlib.pyplot import *
import numpy as np
from collections import defaultdict
import heapq

#Load points from file to correct format (3 lists) 
def loadPoints(file):
    #Load file
    X, Y, Z = [], [], []
    
    with open(file) as f:
        
        for line in f:
            x, y, z = line.split('\t')
            
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))
    
    return X, Y, Z

def drawPoints(X, Y, Z, bx, transp = 0.2):
    #Create figure
    fig = figure()
    ax = axes(projection = '3d')
    ax.set_aspect('equal')

    #Compute sphere scale: 1 pix = 25.4 mm
    scale = 1
    if bx > 0:
        scale = int(bx * bx * 40 * 40)
        
    #Plot points
    ax.scatter(X, Y, Z, s=scale, alpha = transp)

    show()

    
def drawVoxels(x_min, y_min, z_min, dx, dy, dz, V, n_r):
    # Create figure
    fig = figure()
    ax = axes(projection = '3d')
    ax.set_aspect('equal') 
    
    #Create meshgrid
    xedges = linspace(x_min, x_min + dx, n_r+1) #type: ignore
    yedges = linspace(y_min, y_min + dy, n_r+1) #type: ignore
    zedges = linspace(z_min, z_min + dz, n_r+1) #type: ignore
    
    VX, VY, VZ = np.meshgrid(xedges, yedges, zedges,  indexing="ij") 
    
    #Draw voxels
    ax.voxels(VX, VY, VZ, V, edgecolor='k')
    
    show()

def number_of_cells(X):
    #Count number of points in the dataset
    n_b = len(X)
    #Count number of cells in the grid
    n_r = ceil(n_b ** (1/3))
    return int(n_r)

def init_spat_index(X, Y, Z, n_r):
    #Initialize spatial index
    
    #Define the minimum coordinates
    x_min = min(X)
    y_min = min(Y)
    z_min = min(Z)
    
    #Range of the coordinates
    dx = max(X) - x_min
    dy = max(Y) - y_min
    dz = max(Z) - z_min
    
    #Size of the cell
    bx = dx / n_r
    by = dy / n_r
    bz = dz / n_r 
    
    return x_min, y_min, z_min, dx, dy, dz, bx, by, bz
    
def get3Dindex(x, y, z, dx, dy, dz, x_min, y_min, z_min, n_r):
    #Compute 3D voxel indices ix, iy, iz
    #Normalise coordinates to the range of the grid
    xi = (x-x_min) / dx
    yi = (y-y_min) / dy
    zi = (z-z_min) / dz
    
    c = 0.99999999999
    
    #Get indexes ix, iy, iz
    ix = floor(c * n_r * xi)
    iy = floor(c * n_r * yi)
    iz = floor(c * n_r * zi)
    
    return ix, iy, iz

def hash3D(ix, iy, iz, n_r):
    #Generate 1D hash from 3D indices
    j = ix + iy * n_r + iz * n_r ** 2
    
    return j

def voxelize(X, Y, Z, dx, dy, dz, x_min, y_min, z_min, n_r):
    #Voxelize dataset, AI corrected the code here
    #List for storing the voxel index of each point
    J = []
    #Adding indicies to the list
    for i in range(len(X)):
        ix, iy, iz = get3Dindex(X[i], Y[i], Z[i], dx, dy, dz, x_min, y_min, z_min, n_r)
        j = hash3D(ix, iy, iz, n_r)
        J.append(int(j))
    
    return J
        
def createSpatialIDXs(X, Y, Z, dx, dy, dz, x_min, y_min, z_min, n_r):
    #Create two spatial indices for each point, AI corrected the code here
    #Dictionary mapping occupied voxel indices to lists of point indices
    H = defaultdict(list)
    J = voxelize(X, Y, Z, dx, dy, dz, x_min, y_min, z_min, n_r)
    for i, j in enumerate(J):
        key = j
        H[key].append(i)
        
    return H, J

#Find k nearest neighbors of the query point using voxel-based search, AI was used here for understanding and correcting code
#Exclude the query point itself when its index is provided
#Return a list of tuples (x, y, z, distance)
def getNNVoxel(xq, yq, zq, X, Y, Z, H, dx, dy, dz, x_min, y_min, z_min, n_r, k=1, query_index=None):
    
    #Get voxel index of query point
    ix, iy, iz = get3Dindex(xq, yq, zq, dx, dy, dz, x_min, y_min, z_min, n_r)
    
    #Dictionary for storing candidate points and their distances
    candidate_distances = {}
    
    #Size of voxels
    bx = dx / n_r
    by = dy / n_r
    bz = dz / n_r
    
    #Initial search radius
    radius = 0

    #Expand search area
    while radius < n_r:
    
        #Check neighboring voxels
        for ox in range(-radius, radius + 1):
            for oy in range(-radius, radius + 1):
                for oz in range(-radius, radius + 1):
                    
                #Check only the new outer layer
                    if max(abs(ox), abs(oy), abs(oz)) != radius:
                        continue

                #Calculate neighboring voxel indices
                    nix = ix + ox
                    niy = iy + oy
                    niz = iz + oz

                #Check grid boundaries
                    if (nix < 0 or nix >= n_r or
                        niy < 0 or niy >= n_r or
                        niz < 0 or niz >= n_r):
                        continue

                #Convert 3D indices to 1D index
                    jn = hash3D(nix, niy, niz, n_r)
                
                #Add new candidate points and calculate distances
                    for i in H.get(jn, []):
                        
                        #Exclude query point
                        if i == query_index:
                            continue
                        
                        #Calculate distance only once
                        if i not in candidate_distances:

                            ddx = xq - X[i]
                            ddy = yq - Y[i]
                            ddz = zq - Z[i]

                            d = (ddx*ddx + ddy*ddy + ddz*ddz)**0.5

                            #Store candidate point and distance
                            candidate_distances[i] = d
        
        #Select only k nearest candidate points, AI corrected the code here 
        distances = heapq.nsmallest(k,((d, i) for i, d in candidate_distances.items()))
        
        #Stop only if k nearest neighbors are guaranteed, AI was used here for understanding and correcting code
        if len(distances) >= k:

            #Indices of searched area
            ix_low = max(0, ix - radius)
            ix_high = min(n_r - 1, ix + radius)

            iy_low = max(0, iy - radius)
            iy_high = min(n_r - 1, iy + radius)

            iz_low = max(0, iz - radius)
            iz_high = min(n_r - 1, iz + radius)

            #Spatial boundaries of searched area
            xmin_search = x_min + ix_low * bx
            xmax_search = x_min + (ix_high + 1) * bx

            ymin_search = y_min + iy_low * by
            ymax_search = y_min + (iy_high + 1) * by

            zmin_search = z_min + iz_low * bz
            zmax_search = z_min + (iz_high + 1) * bz

            #Distances to boundaries adjacent to unsearched voxels, AI corrected the code here
            boundary_distances = []

            if ix_low > 0:
                boundary_distances.append(xq - xmin_search)

            if ix_high < n_r - 1:
                boundary_distances.append(xmax_search - xq)

            if iy_low > 0:
                boundary_distances.append(yq - ymin_search)

            if iy_high < n_r - 1:
                boundary_distances.append(ymax_search - yq)

            if iz_low > 0:
                boundary_distances.append(zq - zmin_search)

            if iz_high < n_r - 1:
                boundary_distances.append(zmax_search - zq)

            #Minimum distance to unsearched area. Condition to end search, AI corrected the code here
            d_boundary = min(boundary_distances, default=float('inf'))

            #Distance to k-th nearest candidate, AI corrected the code here
            d_k = distances[k - 1][0]

            #Stop if no unsearched point can be closer
            if d_k <= d_boundary:
                break
    
        #Expand search radius
        radius += 1
    
    #Select k nearest points
    nearest_k = distances[:k]
                    
    #Store nearest neighbours
    results_NN = []
    for d, i in nearest_k:
        results_NN.append((X[i], Y[i], Z[i], d))
                    
    return results_NN

def densityVoxel(X, Y, Z, H, dx, dy, dz, x_min, y_min, z_min, n_r):
    
    #Compute spatial density, AI was used here for understanding and correcting code
    density_NN = []
    
    #Compute density through all points
    for i in range(len(X)):
        
        #Find k nearest neighbor
        results_NN = getNNVoxel(X[i], Y[i], Z[i], X, Y, Z, H, dx, dy, dz, x_min, y_min, z_min, n_r, k=1, query_index = i)
        
        #Extract distances of nearest neighbors
        distances = [d for _, _, _, d in results_NN]
        
        #Store nearest neighbor distance
        if distances:
            density_NN.append(distances[0])
        else:
            density_NN.append(float('nan'))

    #Check for NaN values, AI corrected the code here
    if any(np.isnan(d) for d in density_NN):
        return float('nan')
    
    #Calculate average distance and spatial density
    if density_NN:
        d_average = sum(density_NN) / len(density_NN)
    else:
        d_average = float('nan')

    if d_average > 0:
        ro = 1 / d_average**3
    else:
        ro = float('nan')

    return ro
    
def curvatureVoxel(X, Y, Z, H, dx, dy, dz, x_min, y_min, z_min, n_r, k=30):
    
    #Compute curvature, AI was used here for understanding and correcting code
    curvature_NN = []
    
    #Compute curvature for all points
    for i in range(len(X)):
        #Find k nearest neighbors
        results_NN = getNNVoxel(X[i], Y[i], Z[i], X, Y, Z, H, dx, dy, dz, x_min, y_min, z_min, n_r, k=k, query_index=i)
        
        #Extract coordinates of neighboring points, AI corrected the code here
        points = np.array([[x, y, z] for x, y, z, _ in results_NN])
        
        #Check number of neighbors
        if len(points) < k:
            curvature_NN.append(np.nan)
            continue
        
        #Calculate centroid, AI corrected the code here
        centroid = np.mean(points, axis=0)
        
        #Center coordinates, AI corrected the code here
        centered_points = points - centroid
        
        #Calculate covariance matrix, AI corrected the code here
        covariance_matrix = np.cov(centered_points, rowvar=False)
        
        #Compute eigenvalues, AI corrected the code here
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        
        #Replace negative eigenvalues caused by numerical errors with zero, AI corrected the code here
        lambda_c = np.maximum(eigenvalues, 0)
        
        #Calculating kappa from given equation of curvature, AI corrected the code here
        l1 = lambda_c[0] 
        l2 = lambda_c[1]
        l3 = lambda_c[2]
    
        sum_l = l1 + l2 + l3
    
        if sum_l == 0:      
            curvature = 0
        else:
            curvature = l1 / sum_l
        
        curvature_NN.append(curvature) 
    return curvature_NN

#MAIN PROGRAM
#Load points
X, Y, Z = loadPoints('C:/Users/adamk/OneDrive/Dokumenty/PřF_Magistr/Geoinformatika/geoinf_2026/cv4/tree_18.txt')
#X, Y, Z = loadPoints('tree_18.txt')

print('Voxel method')
print('Computing density and curvature of point cloud')

#Start measuring computation time
t_start_grid = process_time()

#Initialize spatial index
n_r = number_of_cells(X)

x_min, y_min, z_min, dx, dy, dz, bx, by, bz = init_spat_index(X, Y, Z, n_r)

#Create spatial indices
H, J = createSpatialIDXs(X, Y, Z, dx, dy, dz, x_min, y_min, z_min, n_r)

#End spatial indexing
t_index_grid = process_time()

#Compute spatial density
ro = densityVoxel(X, Y, Z, H, dx, dy, dz, x_min, y_min, z_min, n_r)

#End density computation
t_density_grid = process_time()

#Compute curvature
curvature_NN = curvatureVoxel(X, Y, Z, H, dx, dy, dz, x_min, y_min, z_min, n_r, k=30)

#End curvature computation
t_end_grid = process_time()

#Calculate computation times
time_index = t_index_grid - t_start_grid
time_density = t_density_grid - t_index_grid
time_curvature = t_end_grid - t_density_grid
time_total = t_end_grid - t_start_grid

#Print results
print("Number of points:", len(X))
print("Spatial indexing:", time_index, "s")
print("Density computation:", time_density, "s")
print("Curvature computation:", time_curvature, "s")
print("Total computation:", time_total, "s")

print("Density:", ro)
print("Number of curvature values:", len(curvature_NN))
print("Minimum curvature:", np.nanmin(curvature_NN))
print("Maximum curvature:", np.nanmax(curvature_NN))

#Create visualisation of the point cloud with curvature values, AI was used here for understanding and correcting code
#Create figure

print("Generating plot for the point cloud with curvature values")

fig = figure()

#Create 3D plot
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

#Color scale is reversed
values = ax.scatter(X, Y, Z, c=curvature_NN, cmap='plasma_r', marker='.')

#Add colorbar
fig.colorbar(values, ax=ax, label='Curvature visualised by color')

#Add title and axis labels
ax.set_title('Point cloud curvature - Voxel method')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#Show plot
show()

#Visualise relationship between computation time and number of points, AI was used here for understanding and correcting code

print("Generating plot for the relationship between computation time and the number of points")

#Lists for storing results
plot_y_time = []
plot_x_numbers = []

#Number of points for testing
plot_x_axis = [100, 2000, 5000, 7000, 10000]

#Measure computation time for different point cloud sizes
for i in plot_x_axis:

    #Check if dataset contains enough points
    if i > len(X):
        continue

    #Select subset of point cloud
    X_plot = X[:i]
    Y_plot = Y[:i]
    Z_plot = Z[:i]

    #Store number of points
    plot_x_numbers.append(i)

    #Start measuring computation time
    time_start_plot = process_time()

    #Initialize spatial index
    n_r_plot = number_of_cells(X_plot)

    x_min_plot, y_min_plot, z_min_plot, dx_plot, dy_plot, dz_plot, bx_plot, by_plot, bz_plot = init_spat_index(X_plot, Y_plot, Z_plot, n_r_plot)

    #Create spatial indices
    H_plot, J_plot = createSpatialIDXs(X_plot, Y_plot, Z_plot, dx_plot, dy_plot, dz_plot, x_min_plot, y_min_plot, z_min_plot, n_r_plot)

    #Compute spatial density
    plot_density = densityVoxel(X_plot, Y_plot, Z_plot, H_plot, dx_plot, dy_plot, dz_plot, x_min_plot, y_min_plot, z_min_plot, n_r_plot)

    #Compute curvature
    plot_curvature = curvatureVoxel(X_plot, Y_plot, Z_plot, H_plot, dx_plot, dy_plot, dz_plot, x_min_plot, y_min_plot, z_min_plot, n_r_plot, k=30)

    #End measuring computation time
    time_end_plot = process_time()

    #Store computation time
    plot_y_time.append(time_end_plot - time_start_plot)

#Plot visualisation
print('X axis:', plot_x_numbers)
print('Y axis:', plot_y_time)

x = np.array(plot_x_numbers)
y = np.array(plot_y_time)

fig = figure()

plot(x, y, linestyle='dotted')
scatter(x, y)

xlabel('Number of points')
ylabel('Processing time (s)')
title('Computation time vs. number of points - Voxel method')
grid(True)

show()

#Visualise relationship between computation time and voxel size, AI was used here for understanding and correcting code

print("Generating plot for the relationship between computation time and voxel size")

#Lists for storing results
plot_y_voxel_time = []
plot_x_voxel_size = []

#Number of voxels per axis for testing
plot_nr_axis = [5, 10, 20, 35, 50]

#Use the same point cloud for all experiments
X_plot = X
Y_plot = Y
Z_plot = Z

#Get processing time for each voxel size
for n_r_plot in plot_nr_axis:

    #Start measuring computation time
    time_start_plot = process_time()

    #Initialize spatial index
    x_min_plot, y_min_plot, z_min_plot, dx_plot, dy_plot, dz_plot, bx_plot, by_plot, bz_plot = init_spat_index(X_plot, Y_plot, Z_plot, n_r_plot)

    #Create spatial indices
    H_plot, J_plot = createSpatialIDXs(X_plot, Y_plot, Z_plot, dx_plot, dy_plot, dz_plot, x_min_plot, y_min_plot, z_min_plot, n_r_plot)

    #Compute spatial density
    plot_density = densityVoxel(X_plot, Y_plot, Z_plot, H_plot, dx_plot, dy_plot, dz_plot, x_min_plot, y_min_plot, z_min_plot, n_r_plot)

    #Compute curvature
    plot_curvature = curvatureVoxel(X_plot, Y_plot, Z_plot, H_plot, dx_plot, dy_plot, dz_plot, x_min_plot, y_min_plot, z_min_plot, n_r_plot, k=30)

    #End measuring computation time
    time_end_plot = process_time()

    #Store computation time
    plot_y_voxel_time.append(time_end_plot - time_start_plot)

    #Calculate representative voxel size
    voxel_size = (bx_plot * by_plot * bz_plot) ** (1/3)

    #Store voxel size
    plot_x_voxel_size.append(voxel_size)

#Print results
print('Voxel sizes:', plot_x_voxel_size)
print('Computation times:', plot_y_voxel_time)

#Convert results to NumPy arrays
x = np.array(plot_x_voxel_size)
y = np.array(plot_y_voxel_time)

#Plot visualisation
fig = figure()

plot(x, y, linestyle='dotted')
scatter(x, y)

xlabel('Voxel size (coordinate units)')
ylabel('Processing time (s)')
title('Computation time vs. voxel size - Voxel method')
grid(True)

show()