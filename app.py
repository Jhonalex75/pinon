from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import math

app = Flask(__name__)

def calculate_gear_parameters(n, pd, phi_d, r_fillet):
    """Calculate gear parameters based on inputs."""
    # Convert angle from degrees to radians
    phi = math.radians(phi_d)
    
    # Basic calculations
    d = n / pd  # Pitch diameter
    p = math.pi / pd  # Circular pitch
    a = 1 / pd  # Addendum
    b = 1.25 / pd  # Dedendum
    c = b - a  # Clearance
    ht = a + b  # Whole depth
    do = d + 2 * a  # Outside diameter
    dr = d - 2 * b  # Root diameter
    db = d * math.cos(phi)  # Base circle diameter
    rb = db / 2  # Base circle radius
    r = d / 2  # Pitch circle radius
    ro = do / 2  # Outside circle radius
    rr = dr / 2  # Root circle radius
    
    # Calculate tooth thickness
    t = math.pi / (2 * pd)  # Tooth thickness at pitch circle
    
    # Return all parameters as a dictionary
    params = {
        "Número de dientes (n)": n,
        "Paso diametral (Pd)": pd,
        "Ángulo de presión (φ)": f"{phi_d}°",
        "Radio del filete (r_fillet)": r_fillet,
        "Diámetro de paso (d)": f"{d:.4f}",
        "Paso circular (p)": f"{p:.4f}",
        "Addendum (a)": f"{a:.4f}",
        "Dedendum (b)": f"{b:.4f}",
        "Holgura (c)": f"{c:.4f}",
        "Altura total (ht)": f"{ht:.4f}",
        "Diámetro exterior (do)": f"{do:.4f}",
        "Diámetro de raíz (dr)": f"{dr:.4f}",
        "Diámetro del círculo base (db)": f"{db:.4f}",
        "Espesor del diente (t)": f"{t:.4f}"
    }
    
    return params, r, rb, ro, rr, phi, r_fillet

def generate_involute_points(rb, theta_max, num_points=100):
    """Generate points for involute curve."""
    theta = np.linspace(0, theta_max, num_points)
    x = rb * (np.cos(theta) + theta * np.sin(theta))
    y = rb * (np.sin(theta) - theta * np.cos(theta))
    return x, y

def generate_addendum_points(ro, theta_start, theta_end, num_points=20):
    """Generate points for addendum circle."""
    theta = np.linspace(theta_start, theta_end, num_points)
    x = ro * np.cos(theta)
    y = ro * np.sin(theta)
    return x, y

def generate_dedendum_points(rr, theta_start, theta_end, num_points=20):
    """Generate points for dedendum circle."""
    theta = np.linspace(theta_start, theta_end, num_points)
    x = rr * np.cos(theta)
    y = rr * np.sin(theta)
    return x, y

def generate_fillet_points(rr, r_fillet, inv_end_x, inv_end_y, num_points=20):
    """Generate points for fillet curve."""
    # This is a simplified version - a proper fillet would need more complex calculations
    theta = np.linspace(0, np.pi/2, num_points)
    x = inv_end_x - r_fillet * np.cos(theta)
    y = inv_end_y + r_fillet * np.sin(theta)
    return x, y

def plot_gear(n, pd, phi_d, r_fillet, plot_type):
    """Generate gear plot based on parameters and plot type."""
    params, r, rb, ro, rr, phi, r_fillet = calculate_gear_parameters(n, pd, phi_d, r_fillet)
    
    plt.figure(figsize=(10, 10))
    
    if plot_type == "addendum":
        # Just plot the addendum circle
        theta = np.linspace(0, 2*np.pi, 100)
        x = ro * np.cos(theta)
        y = ro * np.sin(theta)
        plt.plot(x, y)
        plt.title("Círculo de Addendum")
        
    elif plot_type == "dedendum":
        # Just plot the dedendum circle
        theta = np.linspace(0, 2*np.pi, 100)
        x = rr * np.cos(theta)
        y = rr * np.sin(theta)
        plt.plot(x, y)
        plt.title("Círculo de Dedendum")
        
    elif plot_type == "involute":
        # Calculate involute curve
        # Angle at which involute intersects outside circle
        theta_max = np.sqrt((ro**2 - rb**2) / rb**2)
        x, y = generate_involute_points(rb, theta_max)
        plt.plot(x, y)
        plt.title("Curva Involuta")
        
    elif plot_type == "fillet":
        # Plot a simplified fillet
        theta_max = np.sqrt((ro**2 - rb**2) / rb**2)
        inv_x, inv_y = generate_involute_points(rb, theta_max)
        x, y = generate_fillet_points(rr, r_fillet, inv_x[0], inv_y[0])
        plt.plot(x, y)
        plt.title("Curva del Filete")
        
    elif plot_type == "line":
        # Plot a radial line
        plt.plot([0, r], [0, 0])
        plt.title("Línea Recta")
        
    elif plot_type == "half_tooth":
        # Plot half a tooth profile
        # Involute curve
        theta_max = np.sqrt((ro**2 - rb**2) / rb**2)
        inv_x, inv_y = generate_involute_points(rb, theta_max)
        
        # Mirror the involute for the other side of the tooth
        inv_x_mirror = inv_x
        inv_y_mirror = -inv_y
        
        # Addendum arc
        tooth_angle = 2 * np.pi / n
        add_x, add_y = generate_addendum_points(ro, 0, tooth_angle/2)
        
        # Dedendum arc
        ded_x, ded_y = generate_dedendum_points(rr, 0, tooth_angle/2)
        
        # Fillet curves (simplified)
        fil_x, fil_y = generate_fillet_points(rr, r_fillet, inv_x[0], inv_y[0])
        fil_x_mirror, fil_y_mirror = generate_fillet_points(rr, r_fillet, inv_x_mirror[0], inv_y_mirror[0])
        
        # Plot all components
        plt.plot(inv_x, inv_y, 'b-')
        plt.plot(inv_x_mirror, inv_y_mirror, 'b-')
        plt.plot(add_x, add_y, 'r-')
        plt.plot(ded_x, ded_y, 'g-')
        plt.plot(fil_x, fil_y, 'm-')
        plt.plot(fil_x_mirror, -fil_y_mirror, 'm-')
        
        plt.title("Perfil de Medio Diente")
        
    elif plot_type == "one_tooth":
        # Plot a complete tooth
        # Involute curves
        theta_max = np.sqrt((ro**2 - rb**2) / rb**2)
        inv_x, inv_y = generate_involute_points(rb, theta_max)
        
        # Calculate tooth angle
        tooth_angle = 2 * np.pi / n
        
        # Rotate the involute to create the second side of the tooth
        inv_x2 = inv_x * np.cos(tooth_angle) - inv_y * np.sin(tooth_angle)
        inv_y2 = inv_x * np.sin(tooth_angle) + inv_y * np.cos(tooth_angle)
        
        # Addendum arc
        add_x, add_y = generate_addendum_points(ro, 0, tooth_angle)
        
        # Dedendum arc
        ded_x, ded_y = generate_dedendum_points(rr, 0, tooth_angle)
        
        # Fillet curves (simplified)
        fil_x, fil_y = generate_fillet_points(rr, r_fillet, inv_x[0], inv_y[0])
        fil_x2, fil_y2 = generate_fillet_points(rr, r_fillet, inv_x2[0], inv_y2[0])
        
        # Plot all components
        plt.plot(inv_x, inv_y, 'b-')
        plt.plot(inv_x2, inv_y2, 'b-')
        plt.plot(add_x, add_y, 'r-')
        plt.plot(ded_x, ded_y, 'g-')
        plt.plot(fil_x, fil_y, 'm-')
        plt.plot(fil_x2, fil_y2, 'm-')
        
        plt.title("Perfil de Un Diente")
        
    elif plot_type == "full_gear":
        # Plot the complete gear
        # Calculate tooth angle
        tooth_angle = 2 * np.pi / n
        
        # For each tooth
        for i in range(n):
            # Calculate rotation angle
            rot_angle = i * tooth_angle
            
            # Involute curve
            theta_max = np.sqrt((ro**2 - rb**2) / rb**2)
            inv_x, inv_y = generate_involute_points(rb, theta_max)
            
            # Rotate the involute
            inv_x_rot = inv_x * np.cos(rot_angle) - inv_y * np.sin(rot_angle)
            inv_y_rot = inv_x * np.sin(rot_angle) + inv_y * np.cos(rot_angle)
            
            # Rotate the involute for the other side of the tooth
            inv_x_rot2 = inv_x * np.cos(rot_angle + tooth_angle) - inv_y * np.sin(rot_angle + tooth_angle)
            inv_y_rot2 = inv_x * np.sin(rot_angle + tooth_angle) + inv_y * np.cos(rot_angle + tooth_angle)
            
            # Addendum arc
            add_x, add_y = generate_addendum_points(ro, rot_angle, rot_angle + tooth_angle)
            
            # Dedendum arc
            ded_x, ded_y = generate_dedendum_points(rr, rot_angle, rot_angle + tooth_angle)
            
            # Plot components
            plt.plot(inv_x_rot, inv_y_rot, 'b-')
            plt.plot(inv_x_rot2, inv_y_rot2, 'b-')
            plt.plot(add_x, add_y, 'r-')
            plt.plot(ded_x, ded_y, 'g-')
        
        # Plot pitch circle
        theta = np.linspace(0, 2*np.pi, 100)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        plt.plot(x, y, 'k--')
        
        plt.title("Engranaje Completo")
    
    # Set equal aspect ratio
    plt.axis('equal')
    plt.grid(True)
    
    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return plot_url, params

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    params = {}
    error = None
    
    # Default input values
    inputs = {
        'n': 20,
        'pd': 10,
        'phi_d': 20,
        'r_fillet': 0.05
    }
    
    if request.method == 'POST':
        try:
            # Get form inputs
            n = int(request.form.get('n', 20))
            pd = float(request.form.get('pd', 10))
            phi_d = float(request.form.get('phi_d', 20))
            r_fillet = float(request.form.get('r_fillet', 0.05))
            plot_type = request.form.get('plot_type', 'full_gear')
            
            # Update inputs dictionary
            inputs = {
                'n': n,
                'pd': pd,
                'phi_d': phi_d,
                'r_fillet': r_fillet
            }
            
            # Validate inputs
            if n <= 0:
                raise ValueError("El número de dientes debe ser mayor que cero")
            if pd <= 0:
                raise ValueError("El paso diametral debe ser mayor que cero")
            if phi_d <= 0 or phi_d >= 90:
                raise ValueError("El ángulo de presión debe estar entre 0 y 90 grados")
            if r_fillet < 0:
                raise ValueError("El radio del filete debe ser mayor o igual a cero")
            
            # Generate plot and calculate parameters
            plot_url, params = plot_gear(n, pd, phi_d, r_fillet, plot_type)
            
        except Exception as e:
            error = str(e)
    
    return render_template('index.html', plot_url=plot_url, params=params, inputs=inputs, error=error)

if __name__ == '__main__':
    app.run(debug=True)
