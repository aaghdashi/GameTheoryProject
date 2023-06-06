import nashpy as nash
import numpy as np

A = np.array([[0,7],[2,6]]) # The row player
B = np.array([[0,2],[7,6]]) # The column player
game5 = nash.Game(A,B)
game5

# Find the Nash Equilibrium

equilibria = game5.support_enumeration()

for eq in equilibria:
    print(eq)

