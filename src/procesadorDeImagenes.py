from tkinter import *
from tkinter import filedialog
import cv2
import imutils  
from math import sqrt
from statistics import median
from statistics import mode
from PIL import Image
from PIL import ImageTk
from skimage.measure import shannon_entropy
import numpy as np

def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA, pxstep0=0,pxstep1=0):
    x = pxstep0
    y = pxstep1
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep0

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep1

def elegir_Imagen():
    #Se mandan a llamar a las variables globales
    global nFilas, nColumnas, enFilas, enColumna, path_Image
    #Si los Entry tienen valores enteros positivos
    if enFilas.get()!="" and enColumna.get()!="" and int(enFilas.get())>0 and int(enColumna.get())>0:
        path_Image = filedialog.askopenfilename(filetypes=[#Se especifican los tipos de archivos, para elegir solo imagenes
        ("image", ".jpg"),
        ("image",".jpeg"),
        ("image",".png")])
        #Asignacion de numero de filas y columnas
        nFilas=int(enFilas.get())
        nColumnas=int(enColumna.get())
        #Lectura del archivo
        if len(path_Image)>0:
            global image
            #se lee la imagen de entrada
            image = cv2.imread(path_Image)
            #Se dibuja la rejilla
            altura = image.shape[0]
            ancho = image.shape[1]
            alstep = altura//nFilas
            anchstep = ancho//nColumnas
            draw_grid(image,pxstep0=anchstep,pxstep1=alstep)
            #Para visualizar la imagen de entrada en la GUI
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)#DE BGR A RGB
            im = Image.fromarray(image)
            img = ImageTk.PhotoImage(image=im)
            #Se coloca la imagen en el label
            inputImage.configure(image=img)
            inputImage.image=img

def rgb_to_hsv(r, g, b):
    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    # h, s, v = hue, saturation, value
    cmax = max(r, g, b)    # maximum of r, g, b
    cmin = min(r, g, b)    # minimum of r, g, b
    diff = cmax-cmin       # diff of cmax and cmin.
    # if cmax and cmax are equal then h = 0
    if cmax == cmin: 
        h = 0
    # if cmax equal r then compute h
    elif cmax == r: 
        h = (60 * ((g - b) / diff) + 360) % 360
    # if cmax equal g then compute h
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    # if cmax equal b then compute h
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360
    # if cmax equal zero
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100
    # compute v
    v = cmax * 100
    return h, s, v

def entropiaRGB(lista, pixeles):
    marg, edges = np.histogramdd(lista, bins = 256)
    marg = list(map(lambda p: p/pixeles, marg))
    marg = list(filter(lambda p: p > 0, marg))
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))
    return entropy

def generar_Archivo():
    cont = 1
    global image, nFilas, nColumnas, path_Image
    f = open("imagenProcesada.txt","w")
    #Imagen en escala de grises
    imgGray = cv2.imread (path_Image,0)
    altura = image.shape[0]
    ancho = image.shape[1]
    alstep = altura//nFilas
    anchstep = ancho//nColumnas
    for i in range(0,altura-alstep+1,alstep):
        for j in range(0,ancho-anchstep+1,anchstep):
            pixeles, sum_r, sum_b, sum_g, sum_gray, sum_h, sum_s,sum_v = 0, 0, 0, 0, 0, 0, 0, 0
            rt, gt, bt, gray_t, ht, st, vt = 0, 0, 0, 0, 0, 0, 0
            harray, sarray, varray, rArray, bArray, gArray, grayArray = [], [], [], [], [], [], []
            rArray.clear()
            bArray.clear()
            gArray.clear()
            harray.clear()
            sarray.clear()
            varray.clear()
            grayArray.clear()
            for m in range(i,i+alstep-1):#Media de pixeles
                for n in range(j,j+anchstep-1):
                    b, g, r = image[m,n]
                    gray = imgGray[m,n]
                    h, s, v = rgb_to_hsv(r,g,b)#Se convierte a HSV
                    harray.append(round(h,2))
                    sarray.append(round(s,2))
                    varray.append(round(v,2))
                    rArray.append(r)
                    gArray.append(g)
                    bArray.append(b)
                    grayArray.append(gray)
                    #Sumatoria de los canales
                    rt, gt, bt, ht, st, vt, gray_t = r+rt, g+gt, b+bt, h+ht, s+st, v+vt, gray+gray_t
                    pixeles=pixeles+1
            #Promedio
            rt, gt, bt, ht, st, vt, gray_t = rt/pixeles, gt/pixeles, bt/pixeles, ht/pixeles, st/pixeles, vt/pixeles, gray_t/pixeles
            rt, gt, bt, gray_t = round(rt,2), round(gt,2), round(bt,2), round(gray_t,2) #Redondea los resultados mostrando solo dos decimales.
            ht, st, vt = round(ht,2), round(st,2), round(vt,2)
            for m in range(i,i+alstep-1):
                for n in range(j,j+anchstep-1):
                    b, g, r = image[m,n]
                    gray = imgGray[m,n]
                    h, s, v = rgb_to_hsv(r,g,b)#Se convierte a HSV
                    sum_r=sum_r+(pow(r-rt,2))
                    sum_b=sum_b+(pow(b-bt,2))
                    sum_g=sum_g+(pow(g-gt,2))
                    sum_gray=sum_gray+(pow(gray-gray_t,2))
                    sum_h, sum_s, sum_v = sum_h+(pow(h-ht,2)), sum_s+(pow(s-st,2)), sum_v+(pow(v-vt,2))
            devStandR = sqrt(sum_r/pixeles)
            devStandG = sqrt(sum_g/pixeles)
            devStandB = sqrt(sum_b/pixeles)
            devStandGray = sqrt(sum_gray/pixeles)
            devStandH, devStandS, devStandV = round(sqrt(sum_h/pixeles),2), round(sqrt(sum_s/pixeles),2), round(sqrt(sum_v/pixeles),2)
            devStandR = round(devStandR,2)
            devStandG = round(devStandG,2)
            devStandB = round(devStandB,2)
            devStandGray = round(devStandGray,2)
            cutimageGray = imgGray[i:i+alstep,j:j+anchstep]
            grayEntropy, rEntropy, gEntropy, bEntropy = round(shannon_entropy(cutimageGray),2), round(entropiaRGB(rArray,pixeles),2), round(entropiaRGB(gArray,pixeles),2), round(entropiaRGB(bArray,pixeles),2)
            hmedian, smedian, vmedian, hmode, smode, vmode, rMedian, gMedian, bMedian, grayMedian, rMode, gMode, bMode, grayMode= median(harray), median(sarray), median(varray), mode(harray), mode(sarray), mode(varray), median(rArray), median(gArray), median(bArray), median(grayArray), mode(rArray), mode(gArray), mode(bArray), mode(grayArray)
            caracteristicas = str(rt)+","+str(gt)+","+str(bt)+","+str(rMedian)+","+str(gMedian)+","+str(bMedian)+","+str(rMode)+","+str(gMode)+","+str(bMode)+","+str(devStandR)+","+str(devStandG)+","+str(devStandB)+","+str(gray_t)+","+str(devStandGray)+","+str(grayMedian)+","+str(grayMode)+","+str(ht)+","+str(st)+","+str(vt)+","+str(devStandH)+","+str(devStandS)+","+str(devStandV)+","+str(hmedian)+","+str(smedian)+","+str(vmedian)+","+str(hmode)+","+str(smode)+","+str(vmode)+","+str(grayEntropy)+","+str(rEntropy)+","+str(gEntropy)+","+str(bEntropy)+",I"
            #Escribimos la cadena en el archivo.
            f.write(str(cont)+" "+caracteristicas+"\n")
            #cutimage = image[i:i+alstep,j:j+anchstep]
            #cv2.imshow(str(cont),cutimage)
            image = cv2.putText(image,str(cont),(j ,i + alstep),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(bt,gt,rt),4,cv2.LINE_AA)
            cont=cont+1
        print("Fila")
    f.close() #Cerramos el archivo de texto.
    #Para visualizar la imagen con los numeros recien agregados
    im = Image.fromarray(image)
    img = ImageTk.PhotoImage(image=im)
    #Se coloca la imagen en el label
    inputImage.configure(image=img)
    inputImage.image=img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("imagen Procesada",image)



#Variables globales
image = None
nFilas, nColumnas = 0, 0
path_Image = ""
#Ventana principal
root = Tk()
#Imagen de entrada
inputImage = Label(root)
inputImage.grid(column=0, row=1, rowspan=10)
#Botón para elegir imagen
btn = Button(root, text="Elegir imagen", width=25, command=elegir_Imagen)
btn.grid(column=0,row=0,padx=5,pady=5)
#Boton para generar archivo
btn = Button(root, text="Generar metadatos", width=25, command=generar_Archivo)
btn.grid(column=0,row=12,padx=5,pady=5)
#Espacio de las filas
lblFila = Label(root, text="Numero de filas")
lblFila.grid(column=1,row=1)
enFilas = Entry(root, width=10)
enFilas.grid(column=1,row=2)
#Espacio de las columnas
lblColumna = Label(root, text="Numero de columnas")
lblColumna.grid(column=1,row=3)
enColumna = Entry(root, width=10)
enColumna.grid(column=1,row=4)


root.mainloop()


