# #######################################################
# Programa que aplica K-means a una imagen
#
# Informacion de ayuda sobre k-means en cv2:
# https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html
#
# web de la que he copiado las operaciones para aplicar k-means a una imagen:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
# #######################################################


import cv2          # para gestion de imagenes
import numpy as np  # para operaciones matematicas
import sys          # para leer parametros del terminal
from calculo_mse import mse

print('EJECUCION: python aplicar_kmeans.py <imagen>')

#intento leer del terminal el nombre del fichero que contiene la imagen. Si no se ha indicado,
# se avisa al usuario y se acaba
if len(sys.argv) == 2:
   # copio el nombre del fichero
   figura = sys.argv[1]

   print("procesando la imagen: |"+ figura + "|")

   # leo la imagen 
   img = cv2.imread(figura,cv2.IMREAD_COLOR)

   Z = img.reshape((-1,3))

   # convert to np.float32
   Z = np.float32(Z)

   # define criteria, number of clusters(K) and apply kmeans()
   # la aplicación de k-means concluye cuando se han realizado 10 iteraciones o el error alcanza el valor 1 como maximo
   # MARISA: en nuestro caso seria mejor dejar solo el criterio de las iteraciones
   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
   K = 8  # numero de colores de la paleta cuantizada

   ret,label,center= cv2.kmeans(Z, K, None, criteria, 10, flags=cv2.KMEANS_PP_CENTERS)
   #ret,label,center=cv2.kmeans(Z,K, None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

   # Now convert back into uint8, and make original image
   center = np.uint8(center)
   res = center[label.flatten()]
   res2 = res.reshape((img.shape)) #imagen cuatizada

   #cv2.imshow('res2',res2)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

   # visualizo los colores RGB de la paleta cuantizada
   print('colores de la paleta cuantizada: ')
   print(center)


   # Llamo a la función mse() para calcular el error cuadratico medio com img res2
   mse_calculated = mse(img, res2)

   #vuelco a disco la imagen cuantizada, para poder compararla con otras
   cv2.imwrite('resultado_kmeans.tif',res2)

   #NOTA: para utilizar esto en el algoritmo PSO, podriamos aplicar el calculo del MSE a la imagen original y a res2,
   #con lo que obtendriamos el mse para la paleta de la particula que se ha generado tras aplicar k-means.
   #Es decir, al aplicar k-means a la posicion de una particula obtenemos dos resultados para seguir aplicando el PSO
   # - la paleta mejorada (nueva posicion de la particula)
   # - el fitness de la particula para esta paleta mejorada
   # <<<<<
else:
   print('El programa necesita una imagen para funcionar')
