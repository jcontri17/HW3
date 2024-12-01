#######################################################################################
###-----FRONT MATTER & FUNCTION DEFINITIONS---------------------------------------#####
#######################################################################################



###-----IMPORTS
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.special import eval_hermite
import math




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



###------SCHRODINGER ODE SYSTEM FUNCTION
def schrodinger_focus(x, y, gam, eps):
    phi = y[0]
    phi_prime = y[1]
    dphi_dx = phi_prime
    dphi_prime_dx = (gam * abs(phi**2) + K*x**2 - eps) * phi
    return [dphi_dx, dphi_prime_dx]



###-----SHOOTING METHOD FUNCTION
def shooting_method_focus(gam, eps_guess):
    #Initial Conditions
    phi0 = 0        #phi(-L) = 0
    phi0p = 1e-2  #guess phi'(0) to start

    #Integrate Schrodinger equation using solve_ivp
    sol = solve_ivp(schrodinger_focus, [xstart,xstop], [phi0, phi0p], t_eval=xspan, args=(gam, eps_guess),rtol=tol,atol=tol)
    return sol.y[0] #returns phi(x)



###-----FUNCTION TO SOLVE FOR EIGENVALUES WITH BISECTION METHOD
def eigValFun_focus(gam, eps_min, eps_max, mode):
	eps_mid = (eps_max + eps_min) /2
	psi_x = shooting_method_focus(gam, eps_mid)[-1]
	while abs(psi_x) > tol:
		#print("eps_min =",eps_min,"eps_max =",eps_max,"psi_eps=",psi_x)
		#eps_mid = (eps_max + eps_min) /2
		#psi_x = shooting_method(gam, eps_mid)[-1]
		if (-1)**(mode+1)*psi_x > 0:
			eps_max = eps_mid
		else:
			eps_min = eps_mid
		
		eps_mid = (eps_max + eps_min) /2
		psi_x = shooting_method_focus(gam, eps_mid)[-1]
	return eps_mid



###-----RHS of the Differential Equation for Harm Osc
def harmenOscar(x, y, eps):
    return [y[1], (eps - K * x**2) * y[0]]



###-----EXACT EIGENFUNCTIONS & EIGENVALUES
def exact_eigenfunction(n, x):
    norm = (np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi)))**-1
    hermite_poly = eval_hermite(n, x)
    return norm * np.exp(-x**2 /2 ) * hermite_poly



###-----FUNCTION TO RUN THE SOLVER FOR DIFFERENT METHODS AND TOLERANCES
def run_solver(method, aveStepSizes):
    for tol in tols:
        sol = solve_ivp(
            harmenOscar,
            xspan,
            [phi0, dphi0dx],
            method=method,
            args=(eps,),
            rtol=tol,
            atol=tol,
        )
        stepSizes = np.diff(sol.t)
        aveStepSize = np.mean(stepSizes)
        aveStepSizes.append(aveStepSize)

        #Print step size for each tolerance level
        #print(f"Method: {method}, Tolerance: {tol}, Ave StepSize: {aveStepSize}")





#######################################################################################
###-----MAIN PROGRAM--------------------------------------------------------------#####
#######################################################################################

###____________________________________________________________________________________
###	PART A
###____________________________________________________________________________________

###------PROBLEM SETUP (SAME FOR B AS WELL)
L = 4       			#Domain limit
dx = 0.1    			#size of dx (x-step)
xstart = -L
xstop = L+dx
xspan = np.arange(xstart, xstop, dx)	#array of x-values
K = 1				    #given in problem statement
n_A = 5                 #allows easier usage later in homework
n = n_A				    #first n normalized eigenfunctions(phi_n) and eigenvalues(eps_n)
tol = 1e-4			    #standard tolerance       

eigvalA = np.zeros(n)			        #creates an nx1 array for the eigenvalues
eigfunA = np.zeros((xspan.size,n))	    #creates a 2L/dx x 1 array for the eigenfunction values

eps_start = 0			#Initial guess for lower bound of epsilon in eigenvalue function
eps_offset = 2			#Some offset value to create an initial upper bound for finding the epsilon


plt.figure(figsize=(10,6))
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
plt.show()


###------SAVING THE ANSWERS
print("Part A--------------------------------------------")

A1 = eigfunA
print("A1 size = ", A1.shape)
np.save('A1.npy', A1)

A2 = eigvalA.T
print("A2 size = ", A2.shape)
np.save('A2.npy', A2)

print("Eigenvalues:", eigvalA)





###_____________________________________________________________________________________
###	PART B
###_____________________________________________________________________________________

###-----PARAMETERS
N = len(xspan)		#Number of points in one direction


###-----POTENTIAL FOR QUANTUM HARMONIC OSCILLATOR (K*x^2)
V = K*xspan**2


###-----SPARSE MATRIX FOR HAMILTONIAN (T + V) USING CENTRAL DIFFERENCE FOR 2ND DERIVATIVE OPERATOR
main_diag = (2/dx**2) + V		        #Diagonal terms
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
plt.figure(figsize=(10,6))
for i in range(n):
	eigValB = eigvalB[i]
	label_i = (f"$\\phi_{{{i+1}}}: \t \\epsilon_{{{i+1}}} = {eigValB:.2f}$")
	plt.plot(xspan, eigfunB[:,i], label=label_i)


plt.title("Part B")
plt.xlabel("x")
plt.ylabel("|$\\phi(x)$|")
plt.legend()
plt.grid(True)
plt.show()


###-----ANSWERING PART B
print("Part B--------------------------------------------")

A3 = eigfunB
print("A3 size = ", A3.shape)
np.save('A3.npy', A3)

A4 = eigvalB.T
print("A4 size = ", A4.shape)
np.save('A4.npy', A4)

print("Eigenvalues:", eigvalB)





###_____________________________________________________________________________________
###	PART C
###_____________________________________________________________________________________

###------PROBLEM SETUP
L = 2       			#Domain limit
dx = 0.1    			#size of dx (x-step)
xstart = -L
xstop = L+dx
xspan = np.arange(xstart, xstop, dx)	#array of x-values
K = 1				    #given in problem statement
n = 2				    #first n normalized eigenfunctions(phi_n) and eigenvalues(eps_n)
tol = 1e-4			    #standard tolerance       
gammas = [0.05, -0.05]  #array of gamma values
m = len(gammas)         #gets the number of gamma values for a loop later on





eigvalC = np.zeros((m, n))			#creates an number of m x n array for the eigenvalues
eigfunC = np.zeros((xspan.size,n,m))	#creates a 2L/dx x n x m array for the eigenfunction values
#label_i = np.zeros(n)

g = 0
for gam in gammas:
    eps_start = 0	#Initial guess for lower bound of epsilon in eigenvalue function
    eps_offset = 2	#Some offset value to create an initial upper bound for finding the epsilon
    
    plt.figure(figsize=(10,6))
    for i in range(n):
        eigvalC[g,i] = eigValFun_focus(gam, eps_start, (i+1)*eps_offset, i)

        #Plot the Eigenfunctions
        eigfunC[:,i,g] = normalize(shooting_method_focus(gam, eigvalC[g,i])) #phi(eps) 2L/dx x 1 array
        eigValC = eigvalC[g,i]
        label_i = (f"$\\phi_{{{i+1}}}: \t \\epsilon_{{{i+1}}} = {eigValC:.2f}$")
        plt.plot(xspan, eigfunC[:,i,g], label=label_i)
        eps_start = eigvalC[g,i]		#Sets the next initial guess of epsilon

    ###-----FINISHING PARAMETERS FOR PLOTTING
    plt.title(f"Part C: $\\gamma = {gam}$")
    plt.xlabel("x")
    plt.ylabel("$\\phi(x)$")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    g = g+1


###------ANSWERING THE QUESTION
print("Part C--------------------------------------------")
A5 = eigfunC[:,:,0]
print("A5 size = ", A5.shape)
np.save('A5.npy', A5)

A6 = eigvalC[0,:]
print("A6 size = ", A6.shape)
np.save('A6.npy', A6)
print(A6,A6.shape)

A7 = eigfunC[:,:,1]
print("A7 size = ", A7.shape)
np.save('A7.npy', A7)

A8 = eigvalC[1,:]
print("A8 size = ", A8.shape)
np.save('A8.npy', A8)
print(A8,A8.shape)





###_____________________________________________________________________________________
###	PART D
###_____________________________________________________________________________________

###-----PARAMETERS
L = 2
K = 1
eps = 1
xspan = [-L, L]


###-----INITIAL CONDITIONS
phi0 = 1
dphi0dx = np.sqrt(K * L**2 - eps)


###-----Tolerances to test
tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]


###-----STORE AVERAGE STEP SIZES FOR EACH METHOD
aveStepSizeRK45 = []
aveStepSizeRK23 = []
aveStepSizeRadau = []
aveStepSizeBDF = []




###-----RUN THE SOLVER FOR EACH METHOD
run_solver('RK45', aveStepSizeRK45)
run_solver('RK23', aveStepSizeRK23)
run_solver('Radau', aveStepSizeRadau)
run_solver('BDF', aveStepSizeBDF)



###-----PLOTTING
plt.figure(figsize=(10,6))
plt.loglog(aveStepSizeRK45, tols, 'o-', label='RK45-4th')
plt.loglog(aveStepSizeRK23, tols, 's-', label='RK23-2nd')
plt.loglog(aveStepSizeRadau, tols, '+-', label='Radau')
plt.loglog(aveStepSizeBDF, tols, 'x-', label='BDF')
plt.xlabel('Average Step Size (log scale)')
plt.ylabel('Tolerance (log scale)')
plt.title('Convergence Study for RK45, RK23, Radau, & BDF')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()



###-----GET THE SLOP USING POLYFIT FOR EACH METHOD
slopeRK45 = np.polyfit(np.log(aveStepSizeRK45), np.log(tols), 1)[0]
slopeRK23 = np.polyfit(np.log(aveStepSizeRK23), np.log(tols), 1)[0]
slopeRadau = np.polyfit(np.log(aveStepSizeRadau), np.log(tols), 1)[0]
slopeBDF = np.polyfit(np.log(aveStepSizeBDF), np.log(tols), 1)[0]


###-----PRINT THE SLOPES FOR EACH METHOD
print(f"Slope of the log-log plot for RK45: {slopeRK45:.2f}")
print(f"Slope of the log-log plot for RK23: {slopeRK23:.2f}")
print(f"Slope of the log-log plot for Radau: {slopeRadau:.2f}")
print(f"Slope of the log-log plot for BDF: {slopeBDF:.2f}")


###-----ANSWERING THE QUESTION
print("Part D--------------------------------------------")
slopes = np.zeros((4,1))
slopes = np.array([slopeRK45, slopeRK23, slopeRadau, slopeBDF])
print("Slopes Vector:", slopes)
A9 = slopes.T
np.save('A9.npy', A9)
print(A9.shape)










###_____________________________________________________________________________________
###	PART E
###_____________________________________________________________________________________

###-----CONSTANTS
L = 4
xstart = -L
xstop = L + dx
dx = 0.1
xspan = np.arange(xstart, xstop, dx)	#array of x-values
n = n_A
tol = 1e-4


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

    integrandValsA = np.array([integrand_A(x) for x in xspan])
    integrandValsB = np.array([integrand_B(x) for x in xspan])

    errorA = np.trapz(integrandValsA, xspan)
    errorB = np.trapz(integrandValsB, xspan)

    eigFuns_ae_err.append(np.sqrt(errorA))
    eigFuns_be_err.append(np.sqrt(errorB))

eigFuns_ae_err = np.array(eigFuns_ae_err)
eigFuns_be_err = np.array(eigFuns_be_err)



###-----COMPUTING EIGNEVALUE ERRORS
eigVals_ae_err = []
eigVals_be_err = []

eigVals_ae_err = 100 * np.abs((eigvalA - exact_eigVals) / exact_eigVals)
eigVals_be_err = 100 * np.abs((eigvalB - exact_eigVals) / exact_eigVals)




###-----ANSWERING THE QUESTION
print("Part E--------------------------------------------")

A10 = eigFuns_ae_err.T
np.save('A10.npy', A10)
print(A10.shape)

A11 = eigVals_ae_err
np.save('A11.npy', A11)
print(A11.shape)

A12 = eigFuns_be_err
np.save('A12.npy', A12)
print(A12.shape)

A13 = eigVals_be_err
np.save('A13.npy', A13)
print(A13.shape)

print("Eigenvalue Errors (A):", eigVals_ae_err)
print("Eigenvalue Errors (B):", eigVals_be_err)
print("Eigenfunction Errors (A):", eigFuns_ae_err)
print("Eigenfunction Errors (B):", eigFuns_be_err)













