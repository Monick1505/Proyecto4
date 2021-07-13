# Primero importo las librerías de interés
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import fft

# Primero resulevo la sección 4.1
# Defino una función para importar la imagen


def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y
    retornar un vector de NumPy con las
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)

    return np.array(img)

# Defino una función para codificar los pixeles a bits


def rgb_a_bit(imagen):
    '''Convierte los pixeles de base
    decimal (de 0 a 255) a binaria
    (de 00000000 a 11111111).

    :param imagen: array de una imagen
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = imagen.shape

    # Número total de pixeles
    n_pixeles = x * y * z

    # Convertir la imagen a un vector unidimensional de n_pixeles
    pixeles = np.reshape(imagen, n_pixeles)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))

    return bits_Rx.astype(int)

# Defino una función y simulo un canal ruidoso que afecta la señal


def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx

# Defino una función para pasar de bits a RGB


def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)

# Ahora debo modificar la señal para modular la señal original de BPSK.
# Esta función se debe ajustar para tener una 16-QMA


def qam_mod(bits, fc, mpp):
    '''Un método que simula el esquema de
    modulación digital 16-QAM.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora (información)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits)  # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)  # mpp: muestras por período

    # Aquí como QAM necesita dos portadoras entonces agrego otra
    portadora2 = np.sin(2*np.pi*fc*t_periodo)
    portadora1 = np.cos(2*np.pi*fc*t_periodo)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp)
    senal_Tx = np.zeros(t_simulacion.shape)
    moduladora = np.zeros(t_simulacion.shape)  # (opcional) señal de bits

    # 4. Debo crear un bucle para asignar la forma de onda:
    # Se tiene:
    # A1 = -3, si b1b2 = 00
    # A1 = -1, si b1b2 = 01
    # A1 = 1, si b1b2 = 11
    # A1 = 3, si b1b2 = 10
    # A2 = 3, si b3b4 = 00
    # A2 = 1, si b3b4 = 01
    # A2 = -1, si b3b4 = 11
    # A2 = -3, si b3b4 = 10

    # Creo banderas para saber en cual bit estoy y cual fue el bit anterior
    primerbit = 1
    segundobit = 0
    tercerbit = 0
    cuartobit = 0
    bitanterior = 0

    # Creo un bucle para asignar la forma de onda
    for i, bit in enumerate(bits):
        # Analizo y tomo decisiones si es el primer bit
        if primerbit == 1:
            if bit == 1:
                bitanterior = 1
                primerbit = 0
                segundobit = 1
            if bit == 0:
                bitanterior = 0
                primerbit = 0
                segundobit = 1
        # Analizo y tomo decisiones si es el segundo bit
        if segundobit == 1:
            if (bit == 1) and (bitanterior == 0):
                senal_Tx[i*mpp:(i+1)*mpp] = portadora1 * -1
                moduladora[i*mpp:(i+1)*mpp] = 1
                segundobit = 0
                tercerbit = 1
            if (bit == 1) and (bitanterior == 1):
                senal_Tx[i*mpp:(i+1)*mpp] = portadora1 * 1
                moduladora[i*mpp:(i+1)*mpp] = 1
                segundobit = 0
                tercerbit = 1
            if (bit == 0) and (bitanterior == 1):
                senal_Tx[i*mpp:(i+1)*mpp] = portadora1 * 3
                moduladora[i*mpp:(i+1)*mpp] = 0
                segundobit = 0
                tercerbit = 1
            if (bit == 0) and (bitanterior == 0):
                senal_Tx[i*mpp:(i+1)*mpp] = portadora1 * -3
                moduladora[i*mpp:(i+1)*mpp] = 0
                segundobit = 0
                tercerbit = 1
        # Analizo y tomo decisiones si es el tercer bit
        if tercerbit == 1:
            if bit == 1:
                bitanterior = 1
                tercerbit = 0
                cuartobit = 1
            if bit == 0:
                bitanterior = 0
                tercerbit = 0
                cuartobit = 1
        # Analizo y tomo decisiones si es el cuarto bit
        if cuartobit == 1:
            if (bit == 1) and (bitanterior == 0):
                senal_Tx[i*mpp:(i+1)*mpp] = portadora2 * 1
                moduladora[i*mpp:(i+1)*mpp] = 1
                cuartobit = 0
                primerbit = 1
            if (bit == 1) and (bitanterior == 1):
                senal_Tx[i*mpp:(i+1)*mpp] = portadora2 * -1
                moduladora[i*mpp:(i+1)*mpp] = 1
                cuartobit = 0
                primerbit = 1
            if (bit == 0) and (bitanterior == 1):
                senal_Tx[i*mpp:(i+1)*mpp] = portadora2 * -3
                moduladora[i*mpp:(i+1)*mpp] = 0
                cuartobit = 0
                primerbit = 1
            if (bit == 0) and (bitanterior == 0):
                senal_Tx[i*mpp:(i+1)*mpp] = portadora2 * 3
                moduladora[i*mpp:(i+1)*mpp] = 0
                cuartobit = 0
                primerbit = 1

    # Cuando termona el bucle, sumo las portadoras
    portadora = portadora1 + portadora2

    # 5. Calcular la potencia promedio de la señal modulada
    P_senal_Tx = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)

    return senal_Tx, P_senal_Tx, portadora, moduladora

# Defino una función para demodular


def demodulador(senal_Rx, portadora, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema BPSK. El criterio
    de demodulación se basa en decodificación por
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits (símbolos) en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(senal_Rx.shape)

    # Pseudo-energía de un período de la portadora
    Es = np.sum(portadora**2)

    # Demodulación
    for i in range(N):
        # Producto interno de dos funciones
        producto = senal_Rx[i*mpp:(i+1)*mpp] * portadora
        senal_demodulada[i*mpp:(i+1)*mpp] = producto
        Ep = np.sum(producto)

        # Criterio de decisión por detección de energía
        if Ep > Es*0:
            bits_Rx[i] = 1
        else:
            bits_Rx[i] = 0

    return bits_Rx.astype(int), senal_demodulada

# Por último asigno valores a variables he invoco funciones creadas
# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = -5   # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema BPSK
senal_Tx, Pm, portadora, moduladora = qam_mod(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador(senal_Rx, portadora, mpp)

# 6. Se visualiza la imagen recibida
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10, 6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)

# Visualizar el cambio entre las señales
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(moduladora[0:600], color='r', lw=2)
ax1.set_ylabel('$b(t)$')

# La señal modulada
ax2.plot(senal_Tx[0:600], color='g', lw=2)
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:600], color='b', lw=2)
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='m', lw=2)
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()

# Ahora resuelvo la sección 4.2

# Asigno un tiempo de muestra
time = np.linspace(0, 0.1, 100)

# Por el formato hay 8 funciones de tiempo f(t)
Ft = np.empty((8, len(time)))

# De las funciones se sabe que a puede ser 1, -1, 3 y -3
A = [-3, -1, 1, 3]

plt.figure()

# Creo un bucle y defino una matriz con los posibles valores
for i in A:
    fp = i*np.cos(2*(np.pi)*fc*time)+i*np.sin(2*(np.pi)*fc*time)
    fq = -i*np.cos(2*(np.pi)*fc*time)+i*np.sin(2*(np.pi)*fc*time)
    Ft[i, :] = fp
    Ft[i+1, :] = fq
    plt.plot(time, fp)
    plt.plot(time, fq)

# Ahora se determina el promedio de las 4 realizaciones en cada instante
# Se define una variable para determinar el promedio de las cuatro
# realizaciones en cada instante
prom = [np.mean(Ft[:, i]) for i in range(len(time))]

plt.plot(time, prom, color='g', label='Promedio de realizaciones')
# Se grafica el valor esperado de la señal
vesp = np.mean(senal_Tx)*time
plt.plot(time, vesp, color='c', label='Valor teórico')

# Se decora un poco el gráfico
plt.title('Realizaciones del proceso aleatorio')
plt.xlabel('$time$')
plt.legend()
plt.show()

# Como se puede ver, el gráfico muestra que el promedio de realizaciones y
# el valor teórico esperado tiene el mismo valor. Por esta razón se puede
# decir que la señal tiene ergodicidad

# Ahora resuelvo sección 4.3
# Importo la libreria fft para poder aplicar la transformada de Fourier

# Realizo el proceso es equivalente a la senal_Tx
# Le aplico transformada de Fourier a la señal obtenida
tfsenal = fft(senal_Tx)

# Defino el periodo de mostreo
time_m = (1/fc) / mpp

# Defino para simplicidad la variable muestras
muestras = len(senal_Tx)

# Defino los ejes
frec = np.linspace(0.0, 1.0/(2.0*time_m), muestras//2)


# Genero un gráfico

plt.plot(frec, 2.0/muestras * np.power(np.abs(tfsenal[0:muestras//2]), 2))
# Defino label
plt.title('Densidad espectral de potencia')
plt.xlabel('$time$')
plt.ylabel('senal_Tx')
plt.xlim(0, 20000)
plt.grid()
plt.show()

