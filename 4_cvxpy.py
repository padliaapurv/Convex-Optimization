#read in imgage
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

img = mpimg.imread("flower.png")
img = img[:,:,0:3]
m,n,_ = img.shape

np.random.seed(5)
known_ind = np.where(np.random.rand(m,n) >= 0.90)
# grayscale image
M = 0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2]
# known color values
R_known = img[:,:,0]
G_known = img[:,:,1]
B_known = img[:,:,2]
R_known = R_known[known_ind]
G_known = G_known[known_ind]
B_known = B_known[known_ind]

#Variables
R = cp.Variable((m,n))
G = cp.Variable((m,n))
B = cp.Variable((m,n))

#Constraints
constraints = [ M == 0.299*R + 0.578*G + 0.114*B]
constraints += [R[known_ind] == R_known]
constraints += [B[known_ind] == B_known]
constraints += [R >= 0, G >= 0, B >= 0, R <= 1, G <= 1, B <= 1]

#Objective
obj = cp.Minimize(cp.tv(R,G,B))

#Problem
prob = cp.Problem(obj, constraints)
prob.solve()

#Save function
def save_img(filename, R,G,B):
  img = np.stack((np.array(R),np.array(G),np.array(B)), axis=2)
  # turn off ticks and labels of the figure
  plt.tick_params(
    axis='both', which='both', labelleft='off', labelbottom='off',
    bottom='off', top='off', right='off', left='off'
  )
  fig = plt.imshow(img)


  plt.savefig(filename,bbox_inches='tight',pad_inches=0.)

print("Optimal Value : ", prob.value)

R_given = np.copy(M);
R_given[known_ind] = R_known;
G_given = np.copy(M);
G_given[known_ind] = G_known;
B_given = np.copy(M);
B_given[known_ind] = B_known;
save_img("flower_given.png", R_given, G_given, B_given)

save_img("flower_regenerated.png", R.value, G.value, B.value)
