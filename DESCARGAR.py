import random
import requests
from requests.api import head
#AIzaSyDJD1AiD7RXJN6dmTTfWNtOIGfJI37QMT8
key = "AIzaSyAYXEonrpaCB3gvxKlhPtzsyMxHchN07Cc" #Por razones de seguridad se ha eliminado la clave API utilizada

for i in range (1,31) : #ciclo para descargar múltiples imágenes

    lat_max = 34.032153 #latitud máxima
    lat_min = 34.065757 #latitud mínima
    lat = random.uniform(lat_min,lat_max) #número aleatorio entre máx y min

    lng_max = -118.239126 #longitud máxima
    lng_min = -118.266510 #longitud mínima
    lng = random.uniform(lng_min,lng_max) #número aleatorio entre máx y min

    fov = 60 #el campo de visión coincide con el de los 2 videojuegos utilizados
    hdng = random.uniform(0,360) #dirección de la cámara (0 es norte, 180 es sur, etc.)
    ptch = random.uniform(-20,30) #inclinación (arriba y abajo)

    url = ("https://maps.googleapis.com/maps/api/streetview?size=640x640&source=outdoor&return_error_code=true&location=" +
    str(lat) + "," + str(lng) + "&fov=" + str(fov) + "&heading=" + str(hdng) + "&pitch=" + str(ptch) + "&key=" + key)
    #algunos otros parámetros importantes: se ha seleccionado la resolución máxima, sólo fotografías del exterior 
    #y que no aparezca una imagen si no hay una disponible (para evitar guardar imágenes en blanco)
    r = requests.get(url) #abre la URL, accediendo a la API y obteniendo la fotografía
    
    img = i #nomenclatura de las imágenes

    folder = "LA" 
    with open("DATA1\\STREETVIEW\\" + folder + "\\" + str(img) + ".jpg", "wb") as f: 
        #abre un directorio y guarda el archivo como imagen
        f.write(r.content)