####################################################################################
## This file contains  all useful functions for exploratory analysis of the data  ##
####################################################################################



## Importation librairies:  
from IPython.display import Image, display
import requests




def show_image(image_url) : 
        ''' This function aims to display an image from an URL'''
        
        response = requests.get(image_url)
        if response.status_code == 200:
            display(Image(url=image_url))
        else:
            print("Failed to retrieve image. HTTP Status code:", response.status_code)