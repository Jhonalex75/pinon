# app.py
# Importar las librerías necesarias
from flask import Flask, render_template, request
import numpy as np
import math
import io
import base64

# Usar un backend de Matplotlib que no requiera un GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Inicializar la aplicación Flask
app = Flask(__name__)

def calculate_gear_profile(n, pd, phi_d, r_fillet, plot_type):
    """
    Calcula la geometría y los puntos de trazado para un engranaje recto.
    Esta función es una traducción y refactorización de la lógica del script de MATLAB.

    Args:
        n (int): Número de dientes.
        pd (float): Paso diametral.
        phi_d (float): Ángulo de presión en grados.
        r_fillet (float): Radio del filete.
        plot_type (str): El tipo de gráfico a generar (qué parte resaltar).

    Returns:
        tuple: Una tupla conteniendo:
            - plot_url (str): Datos de la imagen del gráfico en formato Base64.
            - params (dict): Un diccionario con los parámetros calculados del engranaje.
    """

    # --- 1. Calcular parámetros básicos del engranaje ---
    phi = math.radians(phi_d)  # Ángulo de presión en radianes
    d = n / pd                 # Diámetro de paso
    db = d * math.cos(phi)     # Diámetro del círculo base
    do = d + 2 / pd            # Diámetro exterior (addendum)
    dr = d - 2 * 1.25 / pd     # Diámetro de raíz (dedendum)
    tt = math.pi / (2 * pd)    # Espesor del diente en el círculo de paso
    pc = math.pi / pd          # Paso circular

    # --- 2. Calcular las coordenadas del perfil de la involuta ---
    n1 = 10  # Número de puntos para la curva de involuta
    xp = np.zeros(n1)
    yp = np.zeros(n1)
    theta = np.zeros(n1)
    tp = math.pi * d / (2 * n)

    for i in range(n1):
        # El radio varía desde el diámetro exterior hasta el diámetro base
        r = do / 2 - (do - db) * i / (2 * (n1 - 1))
        
        # Asegurarse de que el argumento de acos no esté fuera del rango [-1, 1]
        acos_arg = db / (2 * r)
        if acos_arg > 1.0: acos_arg = 1.0
        if acos_arg < -1.0: acos_arg = -1.0
        
        pha = math.acos(acos_arg)
        
        # Ecuación de la involuta para el espesor del diente
        t = 2 * r * (tp / d + (math.tan(phi) - phi) - (math.tan(pha) - pha))
        theta[i] = t / (2 * r)
        
        # Convertir de coordenadas polares a cartesianas
        xp[i] = r * math.sin(theta[i])
        yp[i] = r * math.cos(theta[i])

    # --- 3. Calcular el círculo de addendum (exterior) ---
    n2 = 5
    xo = np.zeros(n2)
    yo = np.zeros(n2)
    for i in range(n2):
        theta_o = theta[0] * i / (n2 - 1)
        xo[i] = (do / 2) * math.sin(theta_o)
        yo[i] = (do / 2) * math.cos(theta_o)

    # --- 4. Calcular la porción no-involuta (línea recta) ---
    # Entre el círculo base y el círculo de raíz
    n_xr = 3
    xr = np.zeros(n_xr)
    yr = np.zeros(n_xr)
    
    # Ángulo al final de la curva de involuta
    asin_arg = (xp[n1-1] + r_fillet) / (dr / 2)
    if asin_arg > 1.0: asin_arg = 1.0
    if asin_arg < -1.0: asin_arg = -1.0
    theta0 = math.asin(asin_arg)
    
    for i in range(n_xr):
        xr[i] = xp[n1-1]
        yr[i] = yp[n1-1] - (yp[n1-1] - r_fillet - (dr / 2) * math.cos(theta0)) * (i+1) / n_xr

    # --- 5. Calcular el círculo de dedendum (raíz) ---
    n3 = 5
    xro = np.zeros(n3)
    yro = np.zeros(n3)
    for i in range(n3):
        thetar = theta0 + (math.pi / n - theta0) * i / (n3 - 1)
        xro[i] = dr * math.sin(thetar) / 2
        yro[i] = dr * math.cos(thetar) / 2

    # --- 6. Calcular el filete ---
    n4 = 5
    xf = np.zeros(n4)
    yf = np.zeros(n4)
    for i in range(n4):
        angle = i * math.pi / (2 * (n4 - 1))
        xf[i] = xro[0] - r_fillet * math.cos(angle)
        yf[i] = yro[0] + r_fillet * (1 - math.sin(angle))
    
    # --- 7. Ensamblar medio perfil de diente ---
    # El orden es: addendum, involuta, línea recta, filete, dedendum
    c = np.concatenate([xo, xp, xr, xf, xro])
    e = np.concatenate([yo, yp, yr, yf, yro])
    g = np.vstack([c, e])  # Matriz de 2xN con coordenadas

    # --- 8. Reflejar para obtener el diente completo ---
    reflection_matrix = np.array([[-1, 0], [0, 1]])
    ff = reflection_matrix @ g
    
    # Invertir el orden de los puntos reflejados para un trazado continuo
    f = np.fliplr(ff)
    
    # h contiene el perfil completo de un diente
    h = np.hstack([f, g])

    # --- 9. Rotar y ensamblar el engranaje completo ---
    M = np.array([[], []])
    for i in range(n):
        angle = 2 * math.pi * i / n
        rotation_matrix = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)]
        ])
        # NOTA: La matriz de rotación del código MATLAB parece tener un error de signo.
        # -sin en la posición (2,1) es la convención estándar.
        mm = rotation_matrix @ h
        M = np.hstack([M, mm])

    # Cerrar el contorno
    M = np.hstack([M, M[:, 0].reshape(2, 1)])
    
    # --- 10. Generar el gráfico con Matplotlib ---
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Diccionario para seleccionar qué datos trazar
    plot_data = {
        'addendum': (xo, yo),
        'involute': (xp, yp),
        'line': (xr, yr),
        'fillet': (xf, yf),
        'dedendum': (xro, yro),
        'half_tooth': (g[0, :], g[1, :]),
        'one_tooth': (h[0, :], h[1, :]),
        'full_gear': (M[0, :], M[1, :])
    }
    
    # Trazar el contorno base
    base_x, base_y = M[0, :], M[1, :]
    highlight_x, highlight_y = np.array([]), np.array([])

    if plot_type == 'half_tooth':
        base_x, base_y = h[0, :], h[1, :]
        highlight_x, highlight_y = g[0, :], g[1, :]
    elif plot_type == 'one_tooth':
        base_x, base_y = M[0, :], M[1, :]
        highlight_x, highlight_y = h[0, :], h[1, :]
    elif plot_type != 'full_gear':
        base_x, base_y = g[0, :], g[1, :]
        if plot_type in plot_data:
            highlight_x, highlight_y = plot_data[plot_type]
    
    # Dibujar el contorno base en azul punteado
    ax.plot(base_x, base_y, linestyle='--', color='blue', linewidth=2)
    
    # Dibujar la parte resaltada en rojo sólido
    if highlight_x.any() and highlight_y.any():
        ax.plot(highlight_x, highlight_y, linestyle='-', color='red', linewidth=3)
        
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Visualización del Engranaje - {plot_type.replace("_", " ").title()}', fontsize=16)
    ax.set_xlabel("Eje X")
    ax.set_ylabel("Eje Y")
    ax.grid(True)
    
    # Guardar el gráfico en un buffer en memoria
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig) # Cerrar la figura para liberar memoria
    
    # Codificar la imagen en Base64 para mostrarla en HTML
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # --- 11. Recopilar todos los parámetros calculados ---
    params = {
        "Número de Dientes (N)": n,
        "Paso Diametral (Pd)": pd,
        "Ángulo de Presión (φ)": f"{phi_d}°",
        "Radio del Filete": r_fillet,
        "Diámetro de Paso (d)": f"{d:.4f}",
        "Diámetro de Círculo Base (db)": f"{db:.4f}",
        "Diámetro Exterior (do)": f"{do:.4f}",
        "Diámetro de Raíz (dr)": f"{dr:.4f}",
        "Paso Circular (Pc)": f"{pc:.4f}",
        "Addendum (a)": f"{1/pd:.4f}",
        "Dedendum (b)": f"{1.25/pd:.4f}",
        "Profundidad Total (ht)": f"{(1/pd + 1.25/pd):.4f}",
        "Holgura (c)": f"{0.25/pd:.4f}",
        "Espesor del Diente (t)": f"{tt:.4f}"
    }
    
    return plot_url, params

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Renderiza la página principal y maneja el envío del formulario.
    """
    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            n = int(request.form.get('n', 10))
            pd = float(request.form.get('pd', 6.0))
            phi_d = float(request.form.get('phi_d', 20.0))
            r_fillet = float(request.form.get('r_fillet', 0.05))
            plot_type = request.form.get('plot_type', 'full_gear') # Botón presionado

            # Calcular el perfil y generar el gráfico
            plot_url, params = calculate_gear_profile(n, pd, phi_d, r_fillet, plot_type)

            return render_template('index.html', 
                                   plot_url=plot_url, 
                                   params=params, 
                                   inputs={'n': n, 'pd': pd, 'phi_d': phi_d, 'r_fillet': r_fillet})
        except Exception as e:
            # Manejo básico de errores
            return render_template('index.html', error=str(e))
            
    # Para la solicitud GET inicial, mostrar la página con valores por defecto
    return render_template('index.html', inputs={'n': 10, 'pd': 6, 'phi_d': 20, 'r_fillet': 0.05})

if __name__ == '__main__':
    # Ejecutar la aplicación
    # Para producción, se debería usar un servidor WSGI como Gunicorn o Waitress
    app.run(debug=True)

# --------------------------------------------------------------------------
# templates/index.html
# Este archivo debe estar en una carpeta llamada 'templates'
# --------------------------------------------------------------------------
"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diseño de Engranajes con Flask</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .btn-primary {
            @apply bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-75 transition-colors duration-200;
        }
        .btn-secondary {
            @apply bg-gray-200 text-gray-800 font-semibold rounded-lg hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-opacity-75 transition-colors duration-200;
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-800">

    <div class="container mx-auto p-4 md:p-8">
        <header class="text-center mb-8">
            <h1 class="text-4xl font-bold text-gray-900">Calculadora y Visualizador de Engranajes Rectos</h1>
            <p class="text-lg text-gray-600 mt-2">Una herramienta web basada en Python y Flask para el diseño de engranajes</p>
        </header>

        <div class="flex flex-col lg:flex-row gap-8">
            <!-- Columna de Controles -->
            <div class="lg:w-1/3 bg-white p-6 rounded-xl shadow-lg">
                <h2 class="text-2xl font-bold mb-4 border-b pb-2">Parámetros del Engranaje</h2>
                <form action="/" method="post">
                    <div class="space-y-4">
                        <div>
                            <label for="n" class="block text-sm font-medium text-gray-700">Número de Dientes (n)</label>
                            <input type="number" id="n" name="n" value="{{ inputs.n }}" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500" step="1" required>
                        </div>
                        <div>
                            <label for="pd" class="block text-sm font-medium text-gray-700">Paso Diametral (Pd)</label>
                            <input type="number" id="pd" name="pd" value="{{ inputs.pd }}" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500" step="any" required>
                        </div>
                        <div>
                            <label for="phi_d" class="block text-sm font-medium text-gray-700">Ángulo de Presión (°)</label>
                            <input type="number" id="phi_d" name="phi_d" value="{{ inputs.phi_d }}" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500" step="any" required>
                        </div>
                        <div>
                            <label for="r_fillet" class="block text-sm font-medium text-gray-700">Radio del Filete (r_fillet)</label>
                            <input type="number" id="r_fillet" name="r_fillet" value="{{ inputs.r_fillet }}" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500" step="any" required>
                        </div>
                    </div>
                    
                    <h3 class="text-xl font-bold mt-8 mb-4 border-b pb-2">Visualizar Contorno</h3>
                    <div class="grid grid-cols-2 gap-3">
                         <button type="submit" name="plot_type" value="addendum" class="w-full py-2 btn-secondary">Addendum</button>
                         <button type="submit" name="plot_type" value="involute" class="w-full py-2 btn-secondary">Involuta</button>
                         <button type="submit" name="plot_type" value="line" class="w-full py-2 btn-secondary">Línea Recta</button>
                         <button type="submit" name="plot_type" value="fillet" class="w-full py-2 btn-secondary">Filete</button>
                         <button type="submit" name="plot_type" value="dedendum" class="w-full py-2 btn-secondary">Dedendum</button>
                         <button type="submit" name="plot_type" value="half_tooth" class="w-full py-2 btn-secondary">Medio Diente</button>
                    </div>
                     <button type="submit" name="plot_type" value="one_tooth" class="w-full py-2.5 mt-3 btn-primary">Visualizar Un Diente</button>
                     <button type="submit" name="plot_type" value="full_gear" class="w-full py-2.5 mt-3 btn-primary">Visualizar Engranaje Completo</button>
                </form>
            </div>

            <!-- Columna de Resultados -->
            <div class="lg:w-2/3">
                {% if error %}
                    <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg shadow-lg" role="alert">
                        <strong class="font-bold">¡Error!</strong>
                        <span class="block sm:inline">{{ error }}</span>
                    </div>
                {% elif plot_url %}
                    <div class="bg-white p-6 rounded-xl shadow-lg mb-8">
                        <h2 class="text-2xl font-bold mb-4 text-center">Gráfico del Contorno del Engranaje</h2>
                        <div class="flex justify-center">
                            <img src="data:image/png;base64,{{ plot_url }}" alt="Gráfico del engranaje">
                        </div>
                    </div>
                    <div class="bg-white p-6 rounded-xl shadow-lg">
                        <h2 class="text-2xl font-bold mb-4">Resultados del Cálculo</h2>
                        <div class="overflow-x-auto">
                            <table class="min-w-full divide-y divide-gray-200">
                                <thead class="bg-gray-50">
                                    <tr>
                                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Parámetro</th>
                                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Valor</th>
                                    </tr>
                                </thead>
                                <tbody class="bg-white divide-y divide-gray-200">
                                    {% for key, value in params.items() %}
                                    <tr>
                                        <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{{ key }}</td>
                                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{{ value }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                {% else %}
                     <div class="bg-white p-6 rounded-xl shadow-lg flex flex-col items-center justify-center h-full text-center">
                        <svg class="w-16 h-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V7a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                        <h2 class="text-2xl font-bold text-gray-700">Bienvenido</h2>
                        <p class="text-gray-500 mt-2">Introduce los parámetros en el panel de la izquierda y haz clic en un botón de visualización para generar el gráfico y los cálculos del engranaje.</p>
                    </div>
                {% endif %}
            </div>
        </div>
        
        <footer class="text-center mt-12 text-sm text-gray-500">
            <p>Aplicación de Diseño de Engranajes v1.0 - Portado de MATLAB a Python/Flask.</p>
            <p>Basado en el software original de The McGraw-Hill Companies, Inc. (2004).</p>
        </footer>
    </div>
</body>
</html>
"""