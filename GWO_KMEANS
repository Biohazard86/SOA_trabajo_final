# python implementation of Grey wolf optimization (GWO)
# minimizing rastrigin and sphere function
# https://www.geeksforgeeks.org/implementation-of-grey-wolf-optimization-gwo-algorithm/

import random
import math # cos() for Rastrigin
import copy # array-copying convenience
import sys	 # max float
#Para el tratamiento de imagenes:
import numpy as np
from PIL import Image


#-------fitness functions---------

# rastrigin function
def fitness_rastrigin(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value

#sphere function
def fitness_sphere(position):
	fitness_value = 0.0
	for i in range(len(position)):
		xi = position[i]
		fitness_value += (xi*xi);
	return fitness_value;
#-------------------------

#Aplicar kmeans a cada particula de una manera probabilistica
def k_means(dataset, num_clusters, iterations):

    points = dataset_to_list_points(dataset)

    # INICIALIZACIÃ“N: SelecciÃ³n aleatoria de N puntos y creaciÃ³n de los Clusters
    initial = random.sample(points, num_clusters)
    clusters = [Cluster([p]) for p in initial]

    # Inicializamos una lista para el paso de asignaciÃ³n de objetos
    new_points_cluster = [[] for i in range(num_clusters)]

    converge = False
    it_counter = 0
    while (not converge) and (it_counter < iterations):
        # ASIGNACION
        for p in points:
            i_cluster = get_nearest_cluster(clusters, p)
            new_points_cluster[i_cluster].append(p)

        # ACTUALIZACIÃ“N
        for i, c in enumerate(clusters):
            c.update_cluster(new_points_cluster[i])

        # Â¿CONVERGE?
        converge = [c.converge for c in clusters].count(False) == 0

        # Incrementamos el contador
        it_counter += 1
        new_points_cluster = [[] for i in range(num_clusters)]

        print_clusters_status(it_counter, clusters)

    print_results(clusters)

    plot_results(clusters)

# wolf class
class wolf:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]

        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)

            self.fitness = fitness(self.position) # curr fitness



# grey wolf optimization (GWO)
def gwo(fitness, max_iter, n, dim, minx, maxx):
	rnd = random.Random(0)

    #Aqu'i se inserta la imagen en vez de generar los lobos de forma aleatoria

	# create n random wolves
	population = [ wolf(fitness, dim, minx, maxx, i) for i in range(n)]

	# On the basis of fitness values of wolves
	# sort the population in asc order
	population = sorted(population, key = lambda temp: temp.fitness)

	# best 3 solutions will be called as
	# alpha, beta and gaama
	alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])


	# main loop of gwo
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
				X2[j] = beta_wolf.position[j] - A2 * abs(
				C2 - beta_wolf.position[j] - population[i].position[j])
				X3[j] = gamma_wolf.position[j] - A3 * abs(
				C3 - gamma_wolf.position[j] - population[i].position[j])
				Xnew[j]+= X1[j] + X2[j] + X3[j]

			for j in range(dim):
				Xnew[j]/=3.0

			# fitness calculation of new solution
			fnew = fitness(Xnew)

			# greedy selection
			if fnew < population[i].fitness:
				population[i].position = Xnew
				population[i].fitness = fnew

		# On the basis of fitness values of wolves
		# sort the population in asc order
		population = sorted(population, key = lambda temp: temp.fitness)

		# best 3 solutions will be called as
		# alpha, beta and gaama
		alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

		Iter+= 1
	# end-while

	# returning the best solution
	return alpha_wolf.position

#----------------------------
#----------------------------
#----------------------------
#----------------------------


#Abrimos la imagen y la pasamos a una matriz
img = Image.open('test.png').convert('RGBA')    # la abrimos en modo RGBA
img = img.convert('RGB')    # la convertimos a RGB para no tener el canal alfa, el cuarto elemento de la tupla
arr = np.array(img)

# Guardamos la dimension de la imagen. Tendra una dimension de x,y,3
dim = arr.shape

#operaciones>>>




#Esto va al final, cuando hayamos realizado las operaciones:
#Generamos un array de una sola dimension
flat_arr = arr.ravel()

# convert it to a matrix
vector = np.matrix(flat_arr)


# devolvemos el array a su posicion original
arr2 = np.asarray(vector).reshape(dim)

# Volvemos a generar la imagen
img2 = Image.fromarray(arr2, 'RGB') #pasamos el array 2 a una imagen en formato RGB
#img2.save('resultado.png')  # asi la guardamos
img2.show()             # y la mostramos





# Driver code for rastrigin function

print("\nBegin grey wolf optimization on rastrigin function\n")
dim = 2
fitness = fitness_rastrigin


print("Goal is to minimize Rastrigin's function in " + str(dim) + " variables")
print("Function has known min = 0.0 at (", end="")
for i in range(dim-1):
    print("0, ", end="")
print("0)")

num_particles = 50
max_iter = 100

print("Setting num_particles = " + str(num_particles))
print("Setting max_iter = " + str(max_iter))
print("\nStarting GWO algorithm\n")



best_position = gwo(fitness, max_iter, num_particles, dim, -10.0, 10.0)

print("\nGWO completed\n")
print("\nBest solution found:")
print(["%.6f"%best_position[k] for k in range(dim)])
err = fitness(best_position)
print("fitness of best solution = %.6f" % err)

print("\nEnd GWO for rastrigin\n")


print()
print()


# Driver code for Sphere function
print("\nBegin grey wolf optimization on sphere function\n")
dim = 3
fitness = fitness_sphere


print("Goal is to minimize sphere function in " + str(dim) + " variables")
print("Function has known min = 0.0 at (", end="")
for i in range(dim-1):
    print("0, ", end="")
print("0)")

num_particles = 50
max_iter = 100

print("Setting num_particles = " + str(num_particles))
print("Setting max_iter = " + str(max_iter))
print("\nStarting GWO algorithm\n")



best_position = gwo(fitness, max_iter, num_particles, dim, -10.0, 10.0)

print("\nGWO completed\n")
print("\nBest solution found:")
print(["%.6f"%best_position[k] for k in range(dim)])
err = fitness(best_position)
print("fitness of best solution = %.6f" % err)

print("\nEnd GWO for sphere\n")
