# visualize linear and nonlinear mappings in 3D
# exercise in numpy use
# run in psychopy environment
# import
from psychopy import visual,event,gui
import numpy as np
import time

# initialize
size = 0.4
radius_size = 0.01
n_points = 8
n_dim = 3
point = np.empty([n_points,n_dim])
pr = np.empty([n_points,n_dim-1]) ## the projection of point into 2D
for loop in range(2):
    point[0+4*loop,:] = (0,0,loop*size)
    point[1+4*loop,:] = (size,0,loop*size)
    point[2+4*loop,:] = (size,size,loop*size)
    point[3+4*loop,:] = (0,size,loop*size)
    
win = visual.Window()
colormap = ["red","red","green","green","red","red","green","green"]
xormap = ["red","red","green","green","red","green","green","red"]
theta = 0.1 ## transformation angle
transfoZl = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]) ## around Z-axis
transfoZr = np.array([[np.cos(-theta),-np.sin(-theta),0],[np.sin(-theta),np.cos(-theta),0],[0,0,1]]) ## around Z-axis
base = np.transpose(np.array([[1,0.2,0.5],[0,0.5,0]])) ## 2D space spanned by base
proj_matrix = np.dot(np.linalg.inv(np.dot(np.transpose(base),base)),np.transpose(base))
instruction = visual.TextStim(win,text="Press f, j to rotate left, right; q to quit",pos=(0,0.8))

# ask the mapping
info = {"mapping?":"color"}
dlg = gui.DlgFromDict(info)
map = info["mapping?"]
if map=="color":
    relevant_map = colormap
else:
    relevant_map = xormap

# and based on that, initialize the circles
c1 = visual.Circle(win,radius=radius_size,fillColor=relevant_map[0])
c2 = visual.Circle(win,radius=radius_size,fillColor=relevant_map[1])
c3 = visual.Circle(win,radius=radius_size,fillColor=relevant_map[2])
c4 = visual.Circle(win,radius=radius_size,fillColor=relevant_map[3])
c5 = visual.Circle(win,radius=radius_size,fillColor=relevant_map[4])
c6 = visual.Circle(win,radius=radius_size,fillColor=relevant_map[5])
c7 = visual.Circle(win,radius=radius_size,fillColor=relevant_map[6])
c8 = visual.Circle(win,radius=radius_size,fillColor=relevant_map[7])

# project
while True:
    ## draw stuff
    c1.pos = pr[0,:]
    c2.pos = pr[1,:]
    c3.pos = pr[2,:]
    c4.pos = pr[3,:]
    c5.pos = pr[4,:]
    c6.pos = pr[5,:]
    c7.pos = pr[6,:]
    c8.pos = pr[7,:]
        
    instruction.draw()
    c1.draw()
    c2.draw()
    c3.draw()
    c4.draw()
    c5.draw()
    c6.draw()
    c7.draw()
    c8.draw()
    win.flip()
    getkeys = event.getKeys()
    if "f" in getkeys or "j" in getkeys:
        ## rotate cube
        if getkeys[-1] == "f":
            transfo = transfoZl
        else: ## condition above makes sure it must be "j"
            transfo = transfoZr
        for loop in range(n_points):
            point[loop,:] = np.dot(np.transpose(point[loop,:]),transfo)

        ## project cube into 2D
        for loop in range(n_points):
            pr[loop,:] = np.dot(proj_matrix,point[loop,:])
    if "q" in getkeys:
        break

# wrap up
win.close()