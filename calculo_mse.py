# #######################################################
# Programa que calcula el MSE para dos imagenes dadas
# #######################################################


import cv2          # para gestion de imagenes
import numpy as np  # para operaciones matematicas
import sys          # para leer parametros del terminal

def _initial_check(GT,P):
	assert GT.shape == P.shape, "OJO: las dimensiones de ambas imagenes no coinciden " + \
	str(GT.shape) + " and " + str(P.shape)
	if GT.dtype != P.dtype:
		msg = "Supplied images have different dtypes " + \
			str(GT.dtype) + " and " + str(P.dtype)
		warnings.warn(msg)
	

	if len(GT.shape) == 2:
		GT = GT[:,:,np.newaxis]
		P = P[:,:,np.newaxis]

	return GT.astype(np.float64),P.astype(np.float64)

# #######################################################
# calcula el error cuadratico medio (MSE) de dos imagenes
# El ancho y el alto de ambas imagenes debe coincidir. De lo contrario, se produce un error
# GT y P representan ambas imagenes
#
# Si MSE es 0, las imagenes son identicas
# #######################################################
def mse (GT,P):
	"""calculates mean squared error (mse).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.

	:returns:  float -- mse value.
	"""

	GT,P = _initial_check(GT,P)


	return np.mean((GT.astype(np.float64)-P.astype(np.float64))**2)



print('EJECUCION: python calculo_mse.py <imagen1> <imagen2>')

#intento leer del terminal los nombres de los dos ficheros de imagen. Si no se han indicado,
# se avisa al usuario y se acaba
if len(sys.argv) == 3:
   # copio los nombres de ambos ficheros de imagen
   figura1 = sys.argv[1]
   figura2 = sys.argv[2]

   print("procesando las imagenes: |"+ figura1 + "| y |"+ figura2 + "|")

   # leo ambas imagenes 
   FIG1 = cv2.imread(figura1,cv2.IMREAD_COLOR) 
   FIG2 = cv2.imread(figura2,cv2.IMREAD_COLOR)  

   # calculo el MSE y lo muestro
   resul= mse (FIG1, FIG2) # fitness de la particula
   print('MSE:', resul)
else:
   print('El programa necesita dos imagenes de las mismas dimensiones para funcionar')


