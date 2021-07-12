# Solución del proyecto 4, Modelos probabilisticos
Primero importo las librerias necesarias durante toda la solución del proyecto

# Sección 4.1
- Defino una función que importa la imagen dada
- Defino una función para codificar los pixeles a bits
- Defino una función que simula un canal ruidoso para afectar la lectura de la señal
- Defino una función para pasar bits a RGB
- Creo una función para modular usando 16-QAM, para esto realizo un for y agrego varias banderas.
- Defino una función para demodular
- Llamo todas las funciones paso a paso para crear la imagen

# Sección 4.2
- Asigno un tiempo de muestra
- En 16-QAM hay 8 funciones de tiempo, las defino
- Defino los valores que pueden tomar (-1, -3, 1, 3)
- Creo un bucle y defino una matriz con los posibles valores
- Determino el promedio a cada instante he imprimo el gráfico

# Sección 4.3
- Aplico la transformada de Fourier a la función encontrada
- Defino un periodo de muestreo
- Defino los ejes
- Genero el gráfico
