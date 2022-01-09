##########################################################################
#Ejercicio final SOA
#Elsa M.Nanfum Mbasogo Nabiong 
#Sara Hernandez Sanchez
#Marcos Garcia Martin
#David Barrios Portales
###########################################################################

#Librarias
import cv2          # para gestion de imagenes
import numpy as np  # para operaciones matematicas
import sys          # para leer parametros del terminal
import aplicar_kmeans
import array  #para trabajar con arrays
import random #para numeros aleatorios
from calculo_mse import mse
from operator import attrgetter #Para ordenar las particulas
#------------------------------------------------------------

class Rana:
    def __init__(self, num_colores):
        # POSICION DE LA PARTICULA -> creo el array y doy valor inicial
        # creo array de num_colores filas y 2 columnas. Guardo en cada elemento un valor aleatorio entre 0 y 255 
        num_dimensions=8 #dimensiones del problema
        self.posicion  = np.zeros((num_dimensions, 3))
         
        #Doy un valor grande, todavia no lo he calculado 
        self.solucion  = 10000 
        self.fitness = 10000  
        
        #Doy valores aleatorios a las posiciones
        for i in range(0, num_colores):
           self.posicion[i] = np.random.randint(0, 255, 3)

#---------------Funcion que determina la mejor rana (Mejor rana de la poblacion)segun el fitnes de cada una----------
def determina_super_rana(fitness_ranas,num_ranas):
 #determina la rana con mejor fitness. a la hora de llamar a esta funcion, hay que pasarle el self.suarm.fitness y de aqui iremos trabajando
    aux=fitness_ranas[0] #inicialmente ponemos la variable aux que contendra el mejor fitness en posicion 0
    super_rana=0 #inicialmente super rana estara en la posicion 0
    for i in range (num_ranas):
        if(fitness_ranas[i]<aux): #si encontramos un fitness mas pequeño lo guardamos en super_rana
           aux=fitness_ranas[i]
           super_rana=i      
    return super_rana

#------------------Funcion que determina la peor rana------------------------------------
def determina_peor_rana(super_rana,memeplexes,meme,fitness_ranas):
    fit_peor=fitness_ranas[memeplexes[0]] #inicialiamos los valores de la rana o del memeplex
    peor_rana=memeplexes[0]
    fit_mejor=fitness_ranas[memeplexes[0]]
    super_rana=memeplexes[0]
    #recorremos el memeplex
    for j in range(0,meme):
        if(fitness_ranas[memeplexes[j]]<fit_mejor): 
            #si encontramos una rana con mejor fitness lo guardamos en fit_mejor y su indice en mejor_rana
            fit_mejor= fitness_ranas[memeplexes[j]]
            super_rana=memeplexes[j] 
        if(fitness_ranas[memeplexes[j]]>fit_peor): 
            #si encontramos una rana con peor fitness lo guardamos en fit_peor y su indice en peor_rana
            fit_peor=fitness_ranas[memeplexes[j]]
            peor_rana=memeplexes[j] 

    return fit_mejor,super_rana,fit_peor,peor_rana

#------------------------------------------------------------------------------------------------------------------
def calcular_fit(posicion,tam_paleta,iter_kmeans):
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 0.0)
    ret,label,center= cv2.kmeans(Z, tam_paleta, None, criteria, iter_kmeans, flags=cv2.KMEANS_PP_CENTERS, centers=posicion)
    posicion=center 
            
    center = np.uint8(center) 
    res = center[label.flatten()]
    res2 = res.reshape((FIG1.shape)) #imagen cuatizada
            
    #Guardo la imagen resultante de aplicar kmeans
    figu2='figura2.tif'
    cv2.imwrite(figu2,res2)
    #Vuelvo a leer esa imagen
    FIG2 = cv2.imread(figu2,cv2.IMREAD_COLOR)
            
    #Aplico mse para obtener el fitness de cada rana
    fitness=mse(FIG1, FIG2)
    return fitness 
    

#----------------------------------------------------------------------------------------------------------
class SLF:
    def __init__(self):
        #leo la imagen, con imread, y la almaceno en la variable FIG1
        FIG1=cv2.imread(figura1,cv2.IMREAD_COLOR)
        Z = FIG1.reshape((-1,3))
        Z = np.float32(Z)


        #defino las variables necesarias para mi SFL
        num_iters = 5 #iteraciones SLF
        num_ranas = 6 #numero de ranas de la poblacion (tiene que ser par)
        n_variables=3
        tam_paleta= 8 # colores de la paleta (número de dimensiones del problema)
        iter_kmeans= 10 
        d_max=255 #cambio maximo permitido en la posicion (salto de una rana)
        pos_min=0
        pos_max=255

        #-A-Creo la poblacion inicial de ranas--------------------------
        self.swarm = []
        for i in range(0, num_ranas):
              self.swarm.append(Rana(tam_paleta))
        #------------------------------------------------------------------------------------------
        #-B-Calculo el fitness de cada rana-----------------------------
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 0.0)

        for i in range(0, num_ranas):  # PARA CADA RANA
            ret,label,center= cv2.kmeans(Z, tam_paleta, None, criteria, iter_kmeans, flags=cv2.KMEANS_PP_CENTERS, centers=self.swarm[i].posicion)
            self.swarm[i].posicion=center 
            
            center = np.uint8(center) 
            res = center[label.flatten()]
            res2 = res.reshape((FIG1.shape)) #imagen cuatizada
            
            #Guardo la imagen resultante de aplicar kmeans
            figu2='figura2.tif'
            cv2.imwrite(figu2,res2)
            #Vuelvo a leer esa imagen
            FIG2 = cv2.imread(figu2,cv2.IMREAD_COLOR)
            
            #Aplico mse para obtener el fitness de cada rana
            self.swarm[i].fitness= mse(FIG1, FIG2) 
       
            #------------------------------------------------------------------------------------------

        #-C-Ordeno la poblacion por fitness decreciente------------------
        print ('Ranas ordenadas por fitness decreciente:') 
        #Utilizamos una funcion propia de python para ordenar
        self.swarm=sorted(self.swarm, key=attrgetter('fitness'),reverse=True)
        for i in range(0, num_ranas):
            print('Rana: ',(i+1))
            print (self.swarm[i].posicion)
            print (self.swarm[i].solucion)
            print (self.swarm[i].fitness,"\n")

        #-D- Dividimos la población en m memeplexes------------------------------
        m=2
        j=0 #contador de memeplexes
        pos=0 #contador de posicion dentro de cada memeplex
        memplex=[[0 for j in range (num_ranas//2)] for k in range(m)]
        for i in range (num_ranas):
           memplex[j][pos]=i
           j+=1
           if(j>=m):
             j=0
             pos+=1
        #visualizamos los memplexes
        for j in range (m):
           for pos in range (num_ranas//2):
              print('Memeplex:',j,memplex[j][pos])
       
        tam_meme=len(memplex[j]) #guardamos en tam_meme la cantidad de elementos que tiene un memeplex
        
        #-E-Proceso cada memeplex-------------------------
        for i in range(m):
            print('memeplex',i)
            for j in range(0,num_iters): 
                print ('***ITERACION RANA NUMERO ', j)
                #E.1- Procesar cada memeplex – mejor y peor rana

                #Almacenamos en el vector fitness_part los fitness de todas las ranas
                fitness_part=[]
                for f in range (0,num_ranas):
                    fitness_part.append(self.swarm[f].fitness)
            
                #determinamos la mejor y la peor rana
                super_rana=determina_super_rana(fitness_part,num_ranas) #Mejor rana de la poblacion
                fit_mejor,mejor_rana,fit_peor,peor_rana=determina_peor_rana(super_rana,memplex[i],tam_meme,fitness_part)

                #E.2- Procesar cada memeplex- mejorar la peor rana
                #calculamos Di
                for t in range(tam_paleta):#recorremos las dimensiones del problema
                   for d in range (n_variables): #recorremos las variables que tienen cada dimension
                       rnd=random.randint(0,1)
                       #Ec. 2
                       di=rnd*(self.swarm[mejor_rana].posicion-self.swarm[peor_rana].posicion)
                       #comprobamos si el salto a relizar esta dentro del rango d_max
                       if(di[t][d] < -d_max): 
                           di[t][d] =d_max
                       if(di[t][d] > d_max):
                           di[t][d] =d_max
                       
                       #EC.1
                       aux=self.swarm[peor_rana].posicion+di[t]  #almacenamos en aux la posicion de la peor rana y el salto
                       #comprobamos si la posicion esta dentro del rango permitido
                       if(aux[t][d] < pos_min):  
                           aux[t][d]=pos_min
                       if(aux[t][d] > pos_max):
                           aux[t][d]=pos_min
                       #-----------------------------------------------------------------------------------
                       #calculamos el fitness de aux
                       aux_fitness=calcular_fit(aux,tam_paleta,iter_kmeans)
                   
                       #--------------------------------------------------------------------------------------------
                       #luego lo compararemos si tiene mejor fitness que memeplexes[peor_rana][t], lo sustiyuyo
                       if(aux_fitness<self.swarm[peor_rana].fitness):
                           self.swarm[peor_rana].posicion=aux
                           print("Peor rana mejorada A",self.swarm[peor_rana].posicion)
                       else:
                          #Ec. 3
                          rnd=random.randint(0,1)
                          di=rnd*(self.swarm[super_rana].posicion-self.swarm[peor_rana].posicion)
                          #comprobamos si el salto a relizar esta dentro del rango d_max
                          if(di[t][d] < -d_max): 
                              di[t][d] =d_max
                          if(di[t][d] > d_max):
                              di[t][d] =d_max
                          #EC.1
                          aux=self.swarm[peor_rana].posicion+di[t]  #almacenamos en aux la posicion de la peor rana y el salto
                          #print('Posicion candidata',aux)
                          #comprobamos si la posicion esta dentro del rango permitido
                          if(aux[t][d] < pos_min):  
                              aux[t][d]=pos_min
                          if(aux[t][d] > pos_max):
                              aux[t][d]=pos_min
                          #-----------------------------------------------------------------------------------
                          #calculamos el fitness de aux
                          aux_fitness=calcular_fit(aux,tam_paleta,iter_kmeans)
                 
                          #--------------------------------------------------------------------------------------------
                          if(aux_fitness<self.swarm[peor_rana].fitness):
                              self.swarm[peor_rana].posicion=aux
                              print("Peor rana mejorada B",self.swarm[peor_rana].posicion)
                          else: #una nueva solución aleatoria sustituye a self.swarm[peor_rana].posicion
                              aux[t][d]=random.randint(pos_min,pos_max)
                              self.swarm[peor_rana].posicion=aux
                              #print("Solución aleatoria",aux)
                              print("Peor rana mejorada C",self.swarm[peor_rana].posicion)
        #F- Combinar los memeplexes
        grupo=[]
        for k in range (m):
            for j in range(num_ranas//2):
                grupo.append(memplex[k][j])
        print("Combinacion de los memeplexes.\n",grupo)
        print('MSE:',self.swarm[super_rana].fitness)
        #Solucion al problema.Generamos la imagen resultado con la posicion super_nara
        ret,label,center= cv2.kmeans(Z, tam_paleta, None, criteria, 1, flags=cv2.KMEANS_PP_CENTERS, centers=self.swarm[super_rana].posicion)
        
        self.swarm[super_rana].posicion=center 
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((FIG1.shape))

        figu2='RESULTADO.tif'
        cv2.imwrite(figu2,res2)
        
#--------------------------------------------------------------------------
if __name__ == '__main__':
    colors = 255

    #DETERMINO LA IMAGEN SOBRE LA QUE VOY A TRABAJAR 
    #Informo al usuario de como debe llamar al programa (pasando el nombre de una imagen)
    print ('EJECUCION: python image_reduce.py <imagen>')

    #Si he pasado un argumento desde el terminal al ejecutar el programa, lo tomo como nombre de la imagen

    figura1='snowman.tif'

    #Informo al usuario del fichero que voy a procesar
    print("procesando la imagen: |"+ figura1 + "|" )

    

    # DETERMINO EL ANCHO, ALTO Y NÚMERO DE CANALES DE LA IMAGEN
    # abro la imagen (es una imagen en color) 
    FIG1 = cv2.imread(figura1, cv2.IMREAD_COLOR)   


    # determino la altura, la anchura y el número de canales de la imagen
    alto, ancho, c = FIG1.shape


    #muestro al usuario las dimensiones de la imagen
    print ('dimensiones de la imagen: (alto x ancho)', alto, ancho)

    #muestro el numero de canales (3, si es una imagen RGB)
    print ('canales', c)


    #determino el numero de puntos de la imagen
    n_puntos = alto*ancho
    print ('numero total de pixels la imagen: ', n_puntos )


    #COPIO EN UN ARRAY EL VALOR RGB  DE LOS PUNTOS DE LA IMAGEN
    # quedan almacenados como un vector de n_puntos elementos
    Z = FIG1.reshape((-1,3))
    Z = np.float32(Z)

    # muestro al usuario el primer y el ultimo punto
    print ('Valores RGB del primer punto:', Z[0])
    print ('Valores RGB del ultimo punto:', Z[n_puntos-1])



    # necesita 5 parametros: costFunc, x0, bounds, num_particles, maxiter
    # DEDUZCO QUE REPRESENTAN: 
    #   1:funcion objetivo (paso la funcion func1 que has definido más arriba)
    #4: numero de particulas, 5: numero de iteraciones de PSO


    #dimensiones del espacio de solucion. Vamos a tomar 4, por ejemplo
    r =4

    #defino un vector de r elementos a cero
    xA=np.zeros(r)

    # Si se llama con tres argumentos, np.random.randint(a, b, size) devuelve un array de muestras en [a,b) 
    # y de tam. size.
    xB=np.random.randint(0, 10, r)

    # creo array de r filas y 2 columna. Guardo -5 en la primera fila y 5 en la segunda
    limites =np.full((r, 2), -5)
    limites[:,1]=5

    #muestro los dos vectores
    print ("vector xA ", xA)
    print ("vector xB ", xB)
    print ("vector limites ", limites)

    #llamo a mi SLF
    SLF()
