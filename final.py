'''
PASOS A SEGUIR GWO 
1. Inicializar la población de lobos
2.Calcular el fitness de los lobos (función objetivo MSE)
3.Determinar los lobos alfa,beta y delta 
FOR 
    4.Actualizar la posición de cada lobo 
    5.Ajustar el parámetro a (factor de exploración)
    6. Calcular el fitness de los lobos 
        6.1 aplicó kmeans a la posición del lobo con lo que obtengo una nueva posición (paleta)
        6.2 Con esa nueva posición se define la imagen cuantizada
        6.3 Cálculo el mse entre la imagen original y la cuantizada obtenida en el paso previo 
    7.Determino los lobos alfa,beta y delta 
FIN FOR 
8.Devolver mejor solución
que en nuestro caso la solución al problema será una paleta de r colores RGB
'''


from re import X
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from time import time
from PIL import Image
import random
import math # cos() for Rastrigin
import copy # array-copying convenience
import sys	 # max float
#Para el tratamiento de imagenes:
import numpy as np
from PIL import Image
import sklearn 
from sklearn.utils import shuffle



# rastrigin function
def fitness_rastrigin(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value

#sphere function
def fitness_sphere(position):
    fitness_value = KMeans(n_clusters=8, random_state=0).fit(position)

    return fitness_value
#-------------------------


# wolf class
class wolf:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]

        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)

            self.fitness = fitness(self.position) # curr fitness


def gwo(fitness, max_iter, n, dim, minx, maxx, image_array_sample):
    rnd = random.Random(0)
    population = [ wolf(fitness, dim, minx, maxx, i) for i in range(n)]
    population = sorted(population, key = lambda temp: temp.fitness)
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])
    #print(alpha_wolf.position)
    #print(beta_wolf.position)
    #print(gamma_wolf.position)

    


    #FOR
    Iter = 0
    while Iter < max_iter:
        # after every 10 iterations
		# print iteration number and best fitness value so far
        if Iter % 10 == 0 and Iter > 1:
           print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)
		# linearly decreased from 2 to 0
        a = 2*(1 - Iter/max_iter)
		# updating each population member with the help of best three members
        for i in range(n):
            #mostramos el array image_array_sample
            
            KMeans(n_clusters=8, random_state=0).fit(image_array_sample)
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
			2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2*rnd.random(), 2*rnd.random()

            X1 = [0.0 for i in range(dim)]
            X2 = [0.0 for i in range(dim)]
            X3 = [0.0 for i in range(dim)]
            Xnew = [0.0 for i in range(dim)]
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
				C1 - alpha_wolf.position[j] - population[i].position[j])
                #print(X1[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
				C2 - beta_wolf.position[j] - population[i].position[j])
                #print(X2[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
				C3 - gamma_wolf.position[j] - population[i].position[j])
                #print(X3[j])
                #Nueva posicion:
                Xnew[j]+= X1[j] + X2[j] + X3[j]

                for j in range(dim):
                    Xnew[j]/=3.0
                #print(Xnew[j])

            fnew = fitness(Xnew)
            #print(fnew)
            #print(Xnew)
            
            








# MAIN=======================================================================================================================
dim = 2
num_particles = 50
max_iter = 100
fitness = fitness_rastrigin


#Abrimos la imagen y la pasamos a una matriz
img = Image.open('snowman.tif').convert('RGBA')    # la abrimos en modo RGBA
img = img.convert('RGB')    # la convertimos a RGB para no tener el canal alfa, el cuarto elemento de la tupla
#arr = np.array(img)

# Guardamos la dimension de la imagen. Tendra una dimension de x,y,3
#dim = arr.shape
#china = load_sample_image("bosque.jpg")

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
china = np.array(img, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))

print("Fitting model on a small sub-sample of the data")
image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
#print(image_array_sample)
#print(fitness_sphere(image_array_sample))
gwo(fitness, max_iter, num_particles, dim, 0, 1, image_array_sample)





