import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.integrate import trapz




###------PROBLEM SETUP
L = 4       			#Domain limit
dx = 0.1    			#size of dx (x-step)
xstart = -L
xstop = L+dx
xspan = np.arange(xstart, xstop, dx)	#array of x-values
K = 1				#given in problem statement
n_A = 5
n = n_A				#first n normalized eigenfunctions(phi_n) and eigenvalues(eps_n)
tol = 1e-4			#standard tolerance       



###------SCHRODINGER ODE SYSTEM FUNCTION
def schrodinger(x, y, eps):
	phi = y[0]
	phi_prime = y[1]
	dphi_dx = phi_prime
	dphi_prime_dx = (K*x**2 - eps) * phi
	return [dphi_dx, dphi_prime_dx]



###-----SHOOTING METHOD FUNCTION
def shooting_method(eps_guess):
    #Initial Conditions
    phi0 = 0        #phi(-L) = 0
    phi_prime0 = 1  #guess phi'(0) to start

    #Integrate Schrodinger equation using solve_ivp
    sol = solve_ivp(schrodinger, [xstart,xstop], [phi0, phi_prime0], t_eval=xspan, args=(eps_guess,))
    return sol.y[0] #returns phi(x)



###-----FUNCTION TO SOLVE FOR EIGENVALUES WITH BISECTION METHOD
def eigValFun(eps_min, eps_max, mode):
	eps_mid = (eps_max + eps_min) /2
	psi_x = shooting_method(eps_mid)[-1]
	while abs(psi_x) > tol:
		#print("eps_min =",eps_min,"eps_max =",eps_max,"psi_eps=",psi_x)
		#eps_mid = (eps_max + eps_min) /2
		#psi_x = shooting_method(eps_mid)[-1]
		if (-1)**(mode+1)*psi_x > 0:
			eps_max = eps_mid
		else:
			eps_min = eps_mid
		
		eps_mid = (eps_max + eps_min) /2
		psi_x = shooting_method(eps_mid)[-1]
	return eps_mid



###-----NORMALIZE THE EIGENFUNTIONS
def normalize(phi):
	#print(phi)
	norm = np.sqrt(np.trapz(phi**2, xspan))
	return phi/norm









#######################################################################################
###-----MAIN PROGRAM--------------------------------------------------------------#####
#######################################################################################
###____________________________________________________________________________________
###	PART A
###____________________________________________________________________________________
eigvalA = np.zeros(n)			#creates an nx1 array for the eigenvalues
eigfunA = np.zeros((xspan.size,n))	#creates a 2L/dx x 1 array for the eigenfunction values

eps_start = 0				#Initial guess for lower bound of epsilon in eigenvalue function
eps_offset = 2				#Some offset value to create an initial upper bound for finding the epsilon


for i in range(n):
	eigvalA[i] = eigValFun(eps_start, (i+1)*eps_offset, i)
	#print("Eigenvalue",i+1,":", eigvalA[i])

	#Plot the Eigenfunctions
	eigfunA[:,i] = normalize(shooting_method(eigvalA[i])) #phi(eps) 2L/dx x 1 array
	#eigValA = eigvalA[i][0]
	eigValA = eigvalA[i]
	label_i = (f"$\\phi_{{{i+1}}}: \t \\epsilon_{{{i+1}}} = {eigValA:.2f}$")
	plt.plot(xspan, eigfunA[:,i], label=label_i)
	eps_start = eigvalA[i]		#Sets the next initial guess of epsilon


###-----FINISHING PARAMETERS FOR PLOTTING
plt.title("Part A")
plt.xlabel("x")
plt.ylabel("$\\phi(x)$")
plt.legend()
plt.grid(True)
#plt.show()


###------SAVING THE ANSWERS
A1 = eigfunA
print("A1 size = ", A1.shape)
np.save('A1.npy', A1)

A2 = eigvalA.T
print("A2 size = ", A2.shape)
np.save('A2.npy', A2)




###_____________________________________________________________________________________
###	PART B
###_____________________________________________________________________________________
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import pandas as pd


###-----PARAMETERS
N = len(xspan)		#Number of points in one direction


###-----POTENTIAL FOR QUANTUM HARMONIC OSCILLATOR (K*x^2)
V = K*xspan**2

###-----SPARSE MATRIX FOR HAMILTONIAN (T + V) USING CENTRAL DIFFERENCE FOR 2ND DERIVATIVE OPERATOR
main_diag = (2/dx**2) + V		#Diagonal terms
off_diag = (-1/dx**2) * np.ones(N-1)	#Off-diagonal terms


###-----CREATE A SPARSE TRIDIAGONAL HAMILTONIAN MATRIX
H = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csr')


###-----SOLVE FOR FIRST N SMALLEST EIGENVALUES AND EIGENFUNCTIONS
eigvals, eigfuns = eigs(H, k=n, which='SM')		#SM denotes smallest


###-----SORT EIGENVALUES & EIGENFUNCTIONS
sorted_indices = np.argsort(eigvals)
eigvals_first_n = np.real(eigvals[sorted_indices])
eigfuns_first_n = np.real(eigfuns[:, sorted_indices])


###-----NORMALIZE EIGENFUNCTIONS AND TAKE ABSOLUTE VALUE
eigfunB = np.abs(eigfuns_first_n/np.sqrt(dx*np.sum(eigfuns_first_n**2, axis=0)))
eigvalB = np.real(eigvals[sorted_indices])


###-----PLOTTING THE ABSOLUTE VALUE OF EIGENFUNCTIONS
for i in range(n):
	eigValB = eigvalB[i]
	label_i = (f"$\\phi_{{{i+1}}}: \t \\epsilon_{{{i+1}}} = {eigValB:.2f}$")
	plt.plot(xspan, eigfunB[:,i], label=label_i)

plt.title("Part B")
plt.xlabel("x")
plt.ylabel("|$\\phi(x)$|")
plt.legend()
plt.grid(True)
#plt.show()


###-----ANSWERING PART B
A3 = eigfunB
A4 = eigvalB
np.save('A3.npy', A3)
np.save('A4.npy', A4)






###_____________________________________________________________________________________
###	PART C
###_____________________________________________________________________________________




###_____________________________________________________________________________________
###	PART D
###_____________________________________________________________________________________




###_____________________________________________________________________________________
###	PART E
###_____________________________________________________________________________________
from scipy.integrate import quad
from scipy.special import eval_hermite

###-----CONSTANTS
L = 4
xstart = -L
xstop = L + dx
dx = 0.1
xspan = np.arange(xstart, xstop, dx)	#array of x-values
n = n_A
tol = 1e-4

###-----EXACT EIGENFUNCTIONS & EIGENVALUES
def exact_eigenfunction(n, x):
    norm = (np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi)))**-1
    hermite_poly = eval_hermite(n, x)
    return norm * np.exp(-x**2 /2 ) * hermite_poly

exact_eigVals = np.array([2*i + 1 for i in range(n)])



###-----COMPUTE EIGENFUNCTION ERRORS
eigFuns_ae_err = []
eigFuns_be_err = []
exact_eigFuns = []

for i in range(n):
    exact_eigFuns = np.abs(exact_eigenfunction(i, xspan))
    
    #print(exact_eigFuns.shape, xspan.shape)
    
    numA_eigFun = np.abs(eigfunA[:,i])
    numB_eigFun = np.abs(eigfunB[:,i])
    
    #Compute Error Integral
    integrand_A = lambda x_val: (np.abs(exact_eigenfunction(n, x_val)) - np.interp(x_val, xspan, numA_eigFun))**2
    integrand_B = lambda x_val: (np.abs(exact_eigenfunction(n, x_val)) - np.interp(x_val, xspan, numB_eigFun))**2
    print(integrand_A)
    #errorA, _ = quad(integrand_A, xstart, xstop, epsabs=tol, epsrel=tol)
    #errorB, _ = quad(integrand_B, xstart, xstop, epsabs=tol, epsrel=tol)
    errorA = trapz(integrand_A, xspan)
    errorB = trapz(integrand_B, xspan)
    

    eigFuns_ae_err.append(np.sqrt(errorA))
    eigFuns_be_err.append(np.sqrt(errorB))

eigFuns_ae_err = np.array(eigFuns_ae_err)
eigFuns_be_err = np.array(eigFuns_be_err)



###-----COMPUTING EIGNEVALUE ERRORS
eigVals_ae_err = []
eigVals_be_err = []

eigVals_ae_err = 100 * np.abs((eigvalA - exact_eigVals) / exact_eigVals)
eigVals_be_err = 100 * np.abs((eigvalB - exact_eigVals) / exact_eigVals)

#print("exact:",exact_eigFuns, "comp:", eigfunA)



###-----ANSWERING THE QUESTION
A10 = eigFuns_ae_err
np.save('A10.npy', A10)

A11 = eigVals_ae_err
np.save('A11.npy', A11)

A12 = eigFuns_be_err
np.save('A12.npy', A12)

A13 = eigVals_be_err
np.save('A13.npy', A13)

print("Eigenvalue Errors (A):", eigVals_ae_err)
print("Eigenvalue Errors (B):", eigVals_be_err)
print("Eigenfunction Errors (A):", eigFuns_ae_err)
print("Eigenfunction Errors (B):", eigFuns_be_err)













