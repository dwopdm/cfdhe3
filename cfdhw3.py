import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def analytical_solution(x, t):
    """Analytical solution u(x,t) = sin(2π(x-t))"""
    return np.sin(2*np.pi*(x - t))

def lax_friedrichs(u, dx, dt):
    """Lax-Friedrichs scheme"""
    r = dt/dx
    u_new = np.zeros_like(u)
    u_new[1:-1] = 0.5*(u[2:] + u[:-2]) - 0.5*r*(u[2:] - u[:-2])
    u_new[0] = u_new[-2]  # Periodic boundary
    u_new[-1] = u_new[1]  # Periodic boundary
    return u_new

def upwind(u, dx, dt):
    """Upwind scheme"""
    r = dt/dx
    u_new = np.zeros_like(u)
    u_new[1:] = u[1:] - r*(u[1:] - u[:-1])
    u_new[0] = u_new[-2]  # Periodic boundary
    return u_new

def lax_wendroff(u, dx, dt):
    """Lax-Wendroff scheme"""
    r = dt/dx
    u_new = np.zeros_like(u)
    u_new[1:-1] = u[1:-1] - 0.5*r*(u[2:] - u[:-2]) + 0.5*r*r*(u[2:] - 2*u[1:-1] + u[:-2])
    u_new[0] = u_new[-2]  # Periodic boundary
    u_new[-1] = u_new[1]  # Periodic boundary
    return u_new

def compute_error(numerical, analytical):
    """Compute L2 error"""
    return np.sqrt(np.mean((numerical - analytical)**2))

# Grid parameters
L = 3.0  # Domain length
Nx = 100  # Number of spatial points
dx = L/Nx
x = np.linspace(0, L, Nx+1)

# Time parameters
T = 2.0  # Total simulation time
CFL_numbers = [0.1,0.2,0.3,1.1,1.2,1.3]  # Different CFL numbers for stability analysis

# Initial condition
u0 = np.sin(2*np.pi*x)

# Dictionary to store errors for order of accuracy analysis
errors = {
    'Lax-Friedrichs': [],
    'Upwind': [],
    'Lax-Wendroff': []
}

# Stability analysis
plt.figure(figsize=(15, 5))
for CFL in CFL_numbers:
    dt = CFL*dx
    Nt = int(T/dt)
    t = np.linspace(0, T, Nt+1)
    
    # Initialize solutions
    u_lf = u0.copy()
    u_up = u0.copy()
    u_lw = u0.copy()
    
    # Time stepping
    for n in range(Nt):
        u_lf = lax_friedrichs(u_lf, dx, dt)
        u_up = upwind(u_up, dx, dt)
        u_lw = lax_wendroff(u_lw, dx, dt)
        
        # Check for instability
        if np.any(np.abs(u_lf) > 10) or np.any(np.abs(u_up) > 10) or np.any(np.abs(u_lw) > 10):
            print(f"Instability detected for CFL = {CFL}")
            break
    
    plt.subplot(1, 6, CFL_numbers.index(CFL) + 1)
    plt.plot(x, u0, 'k--', label='Initial')
    plt.plot(x, u_lf, label='Lax-Friedrichs')
    plt.plot(x, u_up, label='Upwind')
    plt.plot(x, u_lw, label='Lax-Wendroff')
    plt.plot(x, analytical_solution(x, T), 'k:', label='Analytical')
    plt.title(f'CFL = {CFL}')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Order of accuracy analysis
dx_values = L/np.array([50, 100, 200, 400])
for dx in dx_values:
    Nx = int(L/dx)
    x = np.linspace(0, L, Nx+1)
    u0 = np.sin(2*np.pi*x)
    
    dt = 0.5*dx  # Use stable CFL = 0.5
    Nt = int(T/dt)
    
    u_lf = u0.copy()
    u_up = u0.copy()
    u_lw = u0.copy()
    
    for n in range(Nt):
        u_lf = lax_friedrichs(u_lf, dx, dt)
        u_up = upwind(u_up, dx, dt)
        u_lw = lax_wendroff(u_lw, dx, dt)
    
    analytical = analytical_solution(x, T)
    errors['Lax-Friedrichs'].append(compute_error(u_lf, analytical))
    errors['Upwind'].append(compute_error(u_up, analytical))
    errors['Lax-Wendroff'].append(compute_error(u_lw, analytical))

# Plot convergence rates
plt.figure(figsize=(8, 6))
for method in errors:
    order = np.polyfit(np.log(dx_values), np.log(errors[method]), 1)[0]
    plt.loglog(dx_values, errors[method], 'o-', label=f'{method} (Order ≈ {order:.2f})')
plt.loglog(dx_values, dx_values, 'k--', label='First Order')
plt.loglog(dx_values, dx_values**2, 'k:', label='Second Order')
plt.xlabel('Δx')
plt.ylabel('L2 Error')
plt.title('Convergence Analysis')
plt.legend()
plt.grid(True)
plt.show()