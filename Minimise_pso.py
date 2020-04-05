import numpy as np
from psopy import minimize
import pandas as pd

class Q:
    def setparams(self,alpha=0.1,eta=1,K=0.306173,L=0.388889):
        self.K = K
        self.L = L
        self.alpha = alpha
        self.eta = eta
        
    def Q(self,rho):
        q1 = (self.alpha * self.K**rho) + (1 - self.alpha)*(self.L**rho)
        q = q1 ** (self.eta/rho)
        return q

    def Q1(self,rho):
        q = self.Q(rho)
        if (self.K**self.alpha)*(self.L**(1-self.alpha)) == 0:
            return 0
        q1_1 = q ** (-rho/self.eta) * math.log( (self.K**self.alpha)*(self.L**(1-self.alpha)) )
        if q == 0:
            return 0
        q1 = q1_1 / math.log(q)
        return q1
    
q = Q()

def f(x):
    return q.Q(x[0])


filename = 'icr-nliq-combined.csv'
df = pd.read_csv(filename)
df = df.fillna(0)
# Read from csv and create numpy array.

constraints = (
    {'type': 'ineq', 'fun': lambda x: x[0]},
    {'type': 'stin', 'fun': lambda x: 0.1 - x[0]}
)

# Because tolerance for strict inequalities is 0.0001,
# I'm generating between 0 and 0.9999 and not 1.0.
# Also, I'm pretty certain that even 100 particles is overkill.
x0 = np.random.uniform(0.0, 0.1, (25, 1))
q.setparams(K=0.3061728395, L=0.38888888899999996)
print(f(x0))
results = []

# This is Python's string formatting using the method `str.format`.
record = 'Record: {:05d}    fun: {:.6f}    x: {:.4f}    constr: {:.4f}  {:.4f}'
cost=[]
position=[]
# Iterate over all records.
for i in range(len(df)):

    # Options must be specified this way because the function modifies the
    # dictionary passed to the parameter `options`.
    q.setparams(K=df['ICR'][i],L=df['NLIQ'][i])
    result = minimize(fun=f, x0=x0,constraints=constraints, options = {'g_rate': 0.6, 'l_rate': 0.3, 'max_velocity': 0.9,'stable_iter': 100,'sttol': 1e-4})
    
    # Print a nice pretty report line.
    print(record.format(i + 1, result['fun'], result['x'][0],result['cvec'][0, 0], result['cvec'][0, 1]))
    position.append(result['x'][0])
    cost.append(result['fun'])
    results.append(result)


df['rho'] = position
df['Internationality'] = cost
#df.to_csv('resultV1.csv')
# I haven't written any code to save the results in a file because I'm not
# sure about the formatting required. Each item in `results` has,
#
#       cvec -- constraint vector for `x`,
#       fun -- the value of the function for `x`,
#       message, status, success -- result of operation,
#       nit -- number of iterations,
#       nsit -- number of stable iterations,
#       x -- the global best value.
