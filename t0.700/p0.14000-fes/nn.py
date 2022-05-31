from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
import numpy as np

weight = 10.0
bias = -1.4*weight
grid = np.arange(1.40,1.60,1e-4).reshape(-1,1)
kT = 0.700
files = ["trip0/colvar", "trip1/colvar", "trip2/colvar", "trip3/colvar", "trip4/colvar", "trip5/colvar", "trip6/colvar", "trip7/colvar", "trip8/colvar", "trip9/colvar"]
trips = []

krr = KernelRidge(alpha=1e-3,gamma=1e4,kernel='rbf')

# smooth interpolation of each individual realization
for ifile in files:
    X = np.loadtxt(ifile, skiprows=1, usecols=1).reshape(-1,1)
    y = np.loadtxt(ifile, skiprows=1, usecols=2).reshape(-1,1)
    X = np.append(X,[[0.0]],axis=0)
    y = np.append(y,[[0.0]],axis=0)
    krr.fit(X,y)
    trips.append(krr.predict(grid))

# cumulant formula
alltrips = np.stack(trips)
fes = np.mean(alltrips,axis=0) - np.var(alltrips,ddof=0,axis=0)/(2.0*kT)

# fit an NN to final estimate of FES
X_train, X_test, y_train, y_test = train_test_split(grid*weight+bias, fes, random_state=9541, shuffle=True, test_size=0.1)

nn = MLPRegressor(hidden_layer_sizes=(12), activation='tanh', solver='adam', alpha=1e-3, early_stopping=False, max_iter=10000, random_state=57595, tol=1e-12)
nn.fit(X_train,y_train)
print(nn.score(X_test,y_test))

grid = np.arange(1.40,1.60,1e-3).reshape(-1,1)
preds = nn.predict(grid*weight+bias).reshape(-1,1)

print(nn.coefs_)
print(nn.intercepts_)

np.savetxt('fes.txt', np.vstack((grid.T, preds.T)).T, fmt='%8.3f  %8.3f')
