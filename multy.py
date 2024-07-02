import shutil
import time
import tkinter as tk
from tkinter import HORIZONTAL, Canvas, Scale, filedialog
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageFilter, ImageTk, ImageDraw,ImageGrab,ImageChops
import os 
import threading
import torch
from torch.cuda.amp import autocast
from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline,StableDiffusionInpaintPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from compel import Compel
from googletrans import Translator
import numpy as np
from super_image import EdsrModel, ImageLoader
import queue  # Importa il modulo queue
import cv2
import matplotlib.pyplot as plt
from clip_interrogator import Config, Interrogator
import requests
from tqdm import tqdm
from typing import List
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
token = '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'
modelsdxl1= "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" 
ModelV6B1Inpainting = "Uminosachi/realisticVisionV40_v40VAE-inpainting"
stablediffusion= "runwayml/stable-diffusion-inpainting"
NSFT= "steinhaug/models-nsfw"
PMI= "goldpulpy/UberRealisticPornMergeUrpmV13Inpainting"
#Model name and path settings
path_to_model = ".\\PMI\\"
model_name="urpm.safetensors"

path_to_model2 = ".\\vulva\\"
model_name2="galaxytimemachinesGTM_v5.safetensors"

nuderefine= ".\\vulva\\refined_-inpainting.safetensors"

vulva= ".\\vulva\\airoticartsVulva_10-inpainting.ckpt"
"""
#Download model if not exists (you have to create a folder path_to_model )
if not os.path.exists(f"{path_to_model}{model_name}"):
    print("Downloading model...")
    response = requests.get("https://huggingface.co/goldpulpy/UberRealisticPornMergeUrpmV13Inpainting/resolve/main/UberRealisticPornMergeUrpmV13Inpainting.safetensors", stream=True)

    with open(f"{path_to_model}{model_name}", "wb") as file:
        total_length = int(response.headers.get('content-length'))
        for chunk in tqdm(response.iter_content(chunk_size=1024), total=total_length//1024, unit='KB'):
            if chunk:
                file.write(chunk)
"""

translator = Translator()

points = []
pointsrettangle=[]
last_point = None  # Define last_point here
w = 0
h = 0
img=None
im=0
ragioblur=2
canvasuno=None
beginimage= None

#strumenti
SELECTLAZO=False
SELECTRETANGOLO=False
q = queue.Queue()

# Initialize the Tkinter window first
window = tk.Tk()
window.geometry('1920x1080')
window.title("Multy Inpainting")
cx, cy = 1904,1008  
canvas = Canvas(window, width=cx, height=cy, bg="dark green")
canvas.pack()
salvasfondo= False
textenglish=None

imageface= None
canvasface= None
photoImg2=[]
strumenti= None
imagepath= None
Img= None
xm=0
ym=0
positionImgX=0
positionImgY=0
x=0
y=0
 
 
def traduci_testo(text):
    global translator
    result = translator.translate(text, src="it", dest="en")
    return result.text

def LoraModel():
    global lorarefine,pipeline,textenglish
    if lorarefine.get() == "Phoebe":
        print("phoebe")
        pipeline.load_lora_weights(".\\Lora\\4lyss4m.safetensors", weight_name="4lyss4m.safetensors", adapter_name="4lyss4m")
        textenglish= "(((4lyss4m)))"+ textenglish+ "<lora:4lyss4m:1.0>,"
    elif lorarefine.get() == "Piper":
        print("piper")
        pipeline.load_lora_weights(".\\Lora\\Holly_Marie_Combs_PMv1_Lora.safetensors", weight_name="Holly_Marie_Combs_PMv1_Lora.safetensors", adapter_name="Holly_Marie_Combs_PMv1_Lora")
        textenglish= "(((Holly_Marie_Combs_PMv1_Lora)))"+ textenglish+ "<lora:Holly_Marie_Combs_PMv1_Lora:1.0>,"
    elif lorarefine.get() == "Prue":
        print("prue")
        pipeline.load_lora_weights(".\\Lora\\PrueHalliwell.safetensors", weight_name="PrueHalliwell.safetensors", adapter_name="PrueHalliwell")
        textenglish= "(((PrueHalliwell)))"+ textenglish+ "<lora:PrueHalliwell:1.0>,"
    elif lorarefine.get() == "Paige":
        print("paige")
        pipeline.load_lora_weights(".\\Lora\\PaigeMatthews.safetensors", weight_name="PaigeMatthews.safetensors", adapter_name="PaigeMatthews")
        textenglish= "(((PaigeMatthews)))"+ textenglish+ "<lora:PaigeMatthews:1.0>,"
    elif lorarefine.get() == "Billie":
        print("billie")
        pipeline.load_lora_weights(".\\Lora\\k4l3yc.safetensors", weight_name="k4l3yc.safetensors", adapter_name="k4l3yc")
        textenglish= "(((k4l3yc)))"+ textenglish+ "<lora:k4l3yc:1.0>,"
    elif lorarefine.get() == "perfection_style_SD1.5":
        print("perfection_style_SD1.5")
        pipeline.load_lora_weights(".\\Lora\\perfectionstyleSD15.safetensors", weight_name="perfectionstyleSD15.safetensors", adapter_name="perfectionstyleSD15")
        textenglish= "(((perfectionstyleSD15)))"+ textenglish+ "<lora:perfectionstyleSD15:1.0>,"
    elif lorarefine.get() == "perfection_style":
        print("perfection_style")
        pipeline.load_lora_weights(".\\Lora\\perfectionstyle.safetensors", weight_name="perfectionstyle.safetensors", adapter_name="perfectionstyle")
        textenglish= "(((perfectionstyle)))"+ textenglish+ "<lora:perfectionstyle:1.0>,"
    elif lorarefine.get() == "p*ss*":
        # Il tuo codice qui
        print("p*ss*")
        pipeline.load_lora_weights(".\\Lora\\pussy10.safetensors", weight_name="pussy10.safetensors", adapter_name="pussy10")
        textenglish= "(((pussy10)))"+ textenglish+ "<lora:pussy10:1.0>,"

 
 
def fillgenerate():
    global canvas,cx,cy,photoImg,img,testo,negative,cfg,steps,combobox_reduce,comboupscale,upscalfast,token,modelsdxl1,ModelV6B1Inpainting,combomodel,riniscifaccia,riniscifacciabutton,FaceActor,ragioblur,stablediffusion,canvasuno,beginimage,imageface,textenglish,conditioning
    global tracketa,trackstrength,path_to_model,path_to_model2,model_name,model_name2,vulva,nuderefine,lorarefine,pipeline
    print("fill")
    text_content = testo.get('1.0','end')
    if text_content.strip()=='': 
        text_content='.'
    text = text_content
    image = Image.open(".\\canvasbegin.png")
        
    #riduci immagine
    rid_image= image.resize((cx//2, cy//2))  # Usa una tupla e l'operatore di divisione intera
    mask = Image.open(".\\mask.png")
    #riduci mask
    rid_mask= mask.resize((cx//2, cy//2))  # Usa una tupla e l'operatore di divisione intera
    textenglish = traduci_testo(text)
    
    negative_content = negative.get('1.0', 'end').strip()
    if negative_content:  # Check if negative_content is not empty
        textpromptnegative= traduci_testo(negative_content)
    else:
        textpromptnegative = "(blurry),( duplicate),(deformed),(octane render, render, drawing, anime, bad photo, bad photography:1.3), (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured"
    #'Inpaint_V6','Inpaint_SDXL1.0'
    if combomodel.get()=="Inpaint_V6":
        print("IV6")
        #lorarefine values=['No_lora','Phoebe','Piper','Prue','Paige','Billie','perfection_style_SD1.5','perfection_style','p*ss*']
        pipeline = AutoPipelineForInpainting.from_pretrained(ModelV6B1Inpainting, torch_dtype=torch.float16, use_auth_token=token).to("cuda")
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config) 
        pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))
        LoraModel()
    elif combomodel.get()=="Inpaint_SDXL1.0":
        print("ISDXL")
        pipeline = AutoPipelineForInpainting.from_pretrained(modelsdxl1,torch_dtype=torch.float16, variant="fp16").to("cuda")
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config) 
        pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))
        LoraModel()
    elif combomodel.get()=="stablediffusion":
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(stablediffusion,torch_dtype=torch.float16)
        pipeline.to("cuda")
        pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))
        LoraModel()
    elif combomodel.get()=="NUDE":
        print("NUDE")
        pipeline = StableDiffusionInpaintPipeline.from_single_file(f"{path_to_model}{model_name}",use_safetensors=True,torch_dtype=torch.float16,).to("cuda")
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config) 
        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False
        LoraModel()
    elif combomodel.get()=="NUDE2":
        print("NUDE2")
        pipeline = StableDiffusionInpaintPipeline.from_single_file(f"{path_to_model2}{model_name2}",use_safetensors=True,torch_dtype=torch.float16,).to("cuda")
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config) 
        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False
        LoraModel()
    elif combomodel.get()=="VULVA":
        print("VULVA")
        pipeline = StableDiffusionInpaintPipeline.from_single_file(vulva,use_safetensors=True,torch_dtype=torch.float16,).to("cuda")
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config) 
        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False
        LoraModel()
    elif combomodel.get()=="NUDE_Refine":
        print("NUDE_Refine")
        pipeline = StableDiffusionInpaintPipeline.from_single_file(nuderefine,use_safetensors=True,torch_dtype=torch.float16,).to("cuda")
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config) 
        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False
        LoraModel()
        
        #NUDE_Refine


        
  
    # Espandi token
    compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
    conditioning = compel(textenglish)
    negative_conditioning = compel(textpromptnegative)
    [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])

    with autocast(), torch.inference_mode():
        #falso riduci e risparmia memoria
        #'riduci immagine', 'non riduci immagine'
        if combobox_reduce.get()=='riduci immagine':
            if combomodel.get()=="Inpaint_SDXL1.0":
                print("ISDXL")
                imagegen = pipeline(prompt=textenglish,negative_prompt=textpromptnegative,image=rid_image, mask_image=rid_mask, height=cy//2, width=cx//2, num_inference_steps=int(steps.get()), guidance_scale=float(cfg.get()),strength= float(trackstrength.get()),eta= float(tracketa.get())).images[0]
            elif combomodel.get()=="Inpaint_V6":
                print("IV6")
                imagegen = pipeline(prompt_embeds=conditioning, negative_propmt_embeds=negative_conditioning, image=rid_image, mask_image=rid_mask, height=cy//2, width=cx//2, num_inference_steps=int(steps.get()), guidance_scale=float(cfg.get()), strength= float(trackstrength.get()),eta= float(tracketa.get()),torch_dtype=torch.float16, use_safetensors=False).images[0]
            elif combomodel.get()=="stablediffusion":
               imagegen = pipeline(prompt=textenglish, image=rid_image, mask_image=rid_mask,height=cy//2, width=cx//2, num_inference_steps=int(steps.get()), guidance_scale=float(cfg.get()),strength= float(trackstrength.get()),eta= float(tracketa.get())).images[0]
            elif combomodel.get()=="NUDE":
                print("Nude")
                imagegen = pipeline(prompt_embeds=conditioning, negative_propmt_embeds=negative_conditioning, image=rid_image, mask_image=rid_mask, height=cy//2, width=cx//2, num_inference_steps=int(steps.get()), guidance_scale=float(cfg.get()), strength= float(trackstrength.get()),eta= float(tracketa.get()),torch_dtype=torch.float16, use_safetensors=False).images[0]
            elif combomodel.get()=="NUDE2":
                    print("Nude2")
                    imagegen = pipeline(prompt_embeds=conditioning, negative_propmt_embeds=negative_conditioning, image=rid_image, mask_image=rid_mask, height=cy//2, width=cx//2, num_inference_steps=int(steps.get()), guidance_scale=float(cfg.get()), strength= float(trackstrength.get()),eta= float(tracketa.get()),torch_dtype=torch.float16, use_safetensors=False).images[0]
            elif combomodel.get()=="VULVA":
                    print("VULVA")
                    imagegen = pipeline(prompt_embeds=conditioning, negative_propmt_embeds=negative_conditioning, image=rid_image, mask_image=rid_mask, height=cy//2, width=cx//2, num_inference_steps=int(steps.get()), guidance_scale=float(cfg.get()), strength= float(trackstrength.get()),eta= float(tracketa.get()),torch_dtype=torch.float16, use_safetensors=False).images[0]
        
            elif combomodel.get()=="NUDE_Refine":
                    print("Nude_Refine")
                    imagegen = pipeline(prompt_embeds=conditioning, negative_propmt_embeds=negative_conditioning, image=rid_image, mask_image=rid_mask, height=cy//2, width=cx//2, num_inference_steps=int(steps.get()), guidance_scale=float(cfg.get()), strength= float(trackstrength.get()),eta= float(tracketa.get()),torch_dtype=torch.float16, use_safetensors=False).images[0]
            
            imagegen.save(".\\canvas.png")       
            # 'rifinisci faccia','non rifinisci faccia'
            if riniscifaccia.get()== 'rifinisci faccia':
                #RIFINISCI VOLTO
                print("REFINE VOLTO")
                # Carica il modello pre-addestrato
                piperefine = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
                piperefine = piperefine.to("cuda")
                
                # Carica l'immagine
                img = load_image(".\\canvas.png").convert("RGB") 
                # Applica il modello di raffinamento all'immagine
                img_refined = piperefine(prompt= textenglish, image=img).images
                if os.path.exists(".\\canvas.png"):
                    os.remove(".\\canvas.png")
                # Salva l'immagine raffinata
                img_refined[0].save(".\\canvas.png")
                
            # Upscale
            if comboupscale.get()== 'fast upscale':
                upscalfast= imagegen.resize((cx,cy),Image.LANCZOS)
                upscalfast.save('canvas.png')
                upscalfast.save('canvasInp.png')
                img=upscalfast
            if comboupscale.get()=="X2":
                model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=2).to("cuda").half()
                inputs = ImageLoader.load_image(imagegen).to("cuda").half()  # Convert inputs to Half before inference
                preds = model(inputs)
                preds = preds.float()  # Convert to float before saving
                ImageLoader.save_image(preds, 'canvas.png')
                ImageLoader.save_image(preds, 'canvasInp.png')
            
        #vero non riduci e consuma piu memoria
        elif combobox_reduce.get()=='non riduci immagine':
            if combomodel.get()=="Inpaint_SDXL1.0":
                print("ISDXL")
                imagegen = pipeline(prompt=textenglish,negative_prompt=textpromptnegative,image=image, mask_image=mask, height=cy, width=cx, num_inference_steps=int(steps.get()), guidance_scale=float(cfg.get()),strength= float(trackstrength.get()),eta= float(tracketa.get())).images[0]
            elif combomodel.get()=="Inpaint_V6":
                print("IV6")
                imagegen = pipeline(prompt_embeds=conditioning, negative_propmt_embeds=negative_conditioning, image=image, mask_image=mask, height=cy, width=cx, num_inference_steps=int(steps.get()), guidance_scale=float(cfg.get()), strength= float(trackstrength.get()),eta= float(tracketa.get()),torch_dtype=torch.float16, use_safetensors=False).images[0]
            elif combomodel.get()=="stablediffusion":
                   print("diffuser")
                   pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))
                   imagegen = pipeline(prompt=textenglish, image=image, mask_image=mask,height=cy, width=cx, num_inference_steps=int(steps.get()), guidance_scale=float(cfg.get())).images[0]
            elif combomodel.get()=="NUDE":
                print("Nude")
                imagegen = pipeline(prompt_embeds=conditioning, negative_propmt_embeds=negative_conditioning, image=image, mask_image=mask, height=cy, width=cx, num_inference_steps=int(steps.get()), guidance_scale=float(cfg.get()), strength= float(trackstrength.get()),eta= float(tracketa.get()),torch_dtype=torch.float16, use_safetensors=False).images[0]
            elif combomodel.get()=="NUDE2":
                    print("Nude2")
                    imagegen = pipeline(prompt_embeds=conditioning, negative_propmt_embeds=negative_conditioning, image=image, mask_image=mask, height=cy, width=cx, num_inference_steps=int(steps.get()), guidance_scale=float(cfg.get()), strength= float(trackstrength.get()),eta= float(tracketa.get()),torch_dtype=torch.float16, use_safetensors=False).images[0]
            elif combomodel.get()=="NUDE_Refine":
                    print("Nude_Refine")
                    imagegen = pipeline(prompt_embeds=conditioning, negative_propmt_embeds=negative_conditioning, image=image, mask_image=mask, height=cy, width=cx, num_inference_steps=int(steps.get()), guidance_scale=float(cfg.get()), strength= float(trackstrength.get()),eta= float(tracketa.get()),torch_dtype=torch.float16, use_safetensors=False).images[0]
        
            imagegen.save(".\\canvas.png")
            imagegen.save(".\\'canvasInp.png'.png")
            
            
            #RIFINISCI VOLTO
            print("REFINE VOLTO")
            # Carica il modello pre-addestrato
            piperefine = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
            piperefine = piperefine.to("cuda")
            
            # Carica l'immagine
            img = load_image(".\\canvas.png").convert("RGB") 
            # Applica il modello di raffinamento all'immagine
            img_refined = piperefine(prompt= textenglish, image=img).images
            if os.path.exists(".\\canvas.png"):
                os.remove(".\\canvas.png")
            # Salva l'immagine raffinata
            img_refined[0].save(".\\canvas.png")
            img_refined[0].save(".\\canvasRef.png")
            while True:
                try:
                    img = Image.open(".\\canvas.png")  # Prova ad aprire l'immagine
                    img.verify()  # Verifica che sia un'immagine valida
                    break  # Se l'immagine si apre e verifica senza errori, esce dal ciclo
                except (IOError, SyntaxError):
                        time.sleep(1)  # Aspetta per un secondo e poi riprova
            #FACEACTOR
            if os.path.exists(".\\swapseed\\generato.png"):
               os.remove(".\\swapseed\\generato.png")
               
            if os.path.exists(".\\canvas.png"):
               print("FACE_ACTOR")
               shutil.copyfile(imageface,".\\swapseed\\volto.png")
               shutil.copyfile('.\\canvas.png',".\\swapseed\\generato.png")
               os.chdir(".\\swapseed")
               os.system("python main.py")
               time.sleep(1)
               os.chdir("..")
               if os.path.exists('.\\canvas.png'):
                  os.remove('.\\canvas.png')
               if os.path.exists(".\\swapseed\\generatedimagewithface.png"):
                    shutil.copyfile(".\\swapseed\\generatedimagewithface.png",'.\\canvasFace_actor.png')
                    shutil.move(".\\swapseed\\generatedimagewithface.png",'.\\canvas.png')
            
        img= Image.open('canvas.png')
        # Now, create the PhotoImage object
        photoImg = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, image=photoImg, anchor='nw')
        canvas.update()
    
                
def create_ui(q):
    global SELECTLAZO,SELECTRETANGOLO,lazobutton,retangolobutton,testo,negative,cfg,steps,comboupscale,combobox_reduce,combomodel
    global riniscifaccia,riniscifacciabutton,FaceActor,canvasface,photoImg2,strumenti,textenglish,RidImg,positionImgY,positionImgX,imagepath,Img,xm,ym,cx,cy,img,tracketa,trackstrength,combobackgraund,combomodelenha,lorarefine
    def attiva_lazo():
        global SELECTLAZO, lazobutton,SELECTRETANGOLO
        if SELECTLAZO== False:
            SELECTLAZO= True
            SELECTRETANGOLO=False
            lazobutton.config(bg='green')
            retangolobutton.config(bg='white')
        elif SELECTLAZO== True:
            SELECTLAZO= False
            lazobutton.config(bg='white')
        print(f"lazo {SELECTLAZO} RETTANGOLO {SELECTRETANGOLO}")
    def attiva_retangolo():
        global SELECTRETANGOLO, retangolobutton,SELECTLAZO
        if SELECTRETANGOLO== False:
            SELECTRETANGOLO= True
            SELECTLAZO=False
            retangolobutton.config(bg='green')
            lazobutton.config(bg='white')
            
        elif SELECTRETANGOLO== True:
            SELECTRETANGOLO= False
            retangolobutton.config(bg='white')
        print(f"lazo {SELECTLAZO} RETTANGOLO {SELECTRETANGOLO}")
        
    
        
    strumenti = tk.Tk()
    strumenti.geometry('550x800')
    strumenti.title("Strumenti")
    strumenti.configure(bg="azure")
    # Disabilita il ridimensionamento della finestra
    strumenti.resizable(False, False)

    openbutton = tk.Button(strumenti, text="Open Image", command=openfile)
    openbutton.grid(row=0, column=0,pady=5)

    
    
    
    
    def delmask():
        if os.path.exists(".//mask.png"):
           os.remove(".//mask.png")
   
    def rifinisciF():
        global textenglish, imageface, photoImg, canvas, testo
        if os.path.exists(".\\canvas.png"):
                print("canvas")
                print(f"tex:{textenglish}")
                if textenglish is None:
                    print("textenglidh null")
                    if testo.get('1.0', 'end').strip() == "":
                        print("testo nullo")
                        print("interoga foto")
                        try:
                            # w ,512, h ,x
                            imageint = Image.open(".\\canvas.png").convert('RGB')
                            wi, hi = imageint.size
                            X = int((256 * hi) / wi)
                            imageint = imageint.resize((256, X), Image.LANCZOS)
                            ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
                            interogazione = ci.interrogate(imageint)
                            print(f"Interogazione: {interogazione}")
                            testo.insert('1.0', interogazione)
                            textenglish= interogazione
                        except Exception as errorinterogate:
                            print(f"interogation:{errorinterogate}")
                    else:
                        textenglish = traduci_testo(testo.get('1.0', 'end'))

                # RIFINISCI VOLTO
                print("REFINE VOLTO")
                # Carica il modello pre-addestrato
                piperefine = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
                piperefine = piperefine.to("cuda")

                # Carica l'immagine
                img = load_image(".\\canvas.png").convert("RGB")
                imx, imy = img.size
                img = img.resize((imx // 2, imy // 2), Image.BICUBIC)

                # Applica il modello di raffinamento all'immagine
                img_refined = piperefine(prompt=textenglish, image=img).images
                if os.path.exists(".\\canvas.png"):
                    os.remove(".\\canvas.png")
                img_refined[0] = img_refined[0].resize((imx, imy), Image.BICUBIC)

                # Salva l'immagine raffinata
                img_refined[0].save(".\\canvas.png")
                img_refined[0].save(".\\canvasRef.png")
                photoImg = ImageTk.PhotoImage(img_refined[0])  # Salva l'oggetto PhotoImage in una variabile globale
                canvas.create_image(0, 0, image=photoImg, anchor='nw')
                canvas.update()
            

                
    
    def FaceActorf():
        global imageface,photoImg,canvas
        print(f"imageface: {imageface}")
        if imageface is not None:
            try:
                if os.path.exists(".\\swapseed\\generato.png"):
                        os.remove(".\\swapseed\\generato.png")     
                print("Face Actor")
                #FACEACTOR
                if os.path.exists(".\\canvas.png"):
                    gen= Image.open(".\\canvas.png")
                    xmi,ymi= gen.size
                    gen= gen.resize((xmi//2,ymi//2),Image.BICUBIC)
                    gen.save(".\\swapseed\\generato.png")
                    print("FACE_ACTOR")
                    shutil.copyfile(imageface,".\\swapseed\\volto.png")
                    os.chdir(".\\swapseed")
                    os.system("python main.py")
                    time.sleep(1)
                    os.chdir("..")
                    if os.path.exists('.\\canvas.png'):
                        os.remove('.\\canvas.png')
                    if os.path.exists(".\\swapseed\\generatedimagewithface.png"):
                        gen= Image.open(".\\swapseed\\generatedimagewithface.png")
                        gen= gen.resize((xmi,ymi),Image.BICUBIC)
                    if os.path.exists('.\\canvas.png'):
                        os.remove('.\\canvas.png')  # Rimuovi il file usando il percorso del file
                    gen.save('.\\canvas.png')
                    gen.save('.\\canvasFace_actor.png')
                    photoImg = ImageTk.PhotoImage(gen)  # Salva l'oggetto PhotoImage in una variabile globale
                    canvas.create_image(0,0,image=photoImg,anchor='nw')
                    canvas.update()
            except Exception as errorefaceactor:
                    print(f"Errore Face Actor: {errorefaceactor}")
            
    
    
    def rilevavestitiF():
        global img,x,y,cx,cy,xm,ym
        print("rileva vestiti")
        from transformers import pipeline
        from PIL import Image
        import numpy as np
        from scipy import ndimage

        sfondo = Image.new('RGB', (cx, cy), (0, 100, 0))
        sfondoblack = Image.new('RGB', (cx, cy), (0, 0, 0))

        xm,ym= img.size    
        if  xm< 1904:
            x = cx//2 - img.width//2
        else:
            x=0
        if ym< 1008:
            y = cy//2 - img.height//2
        else:
            y=0
        sfondo.paste(img.resize((xm,ym),Image.BICUBIC), (x, y))
        sfondo.save('canvasbegin.png')

        clothes= ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf","swimsuit", "bra", "underwear", "briefs", "thong"]
        segmenter = pipeline(model="sayeed99/segformer_b3_clothes")
        segments = segmenter(img)

        mask_list = []
        for s in segments:
            if(s['label'] in clothes):
                mask_list.append(s['mask'])

        final_mask = np.array(mask_list[0])
        for mask in mask_list:
            current_mask = np.array(mask)
            final_mask = np.maximum(final_mask, current_mask)

        # Fill holes in the final mask
        final_mask = ndimage.binary_fill_holes(final_mask > 0)

        # Dilate the final mask by 10 pixels
        final_mask = ndimage.binary_dilation(final_mask, iterations=10)

        final_mask = Image.fromarray(final_mask.astype(np.uint8) * 255)
        sfondoblack.paste(final_mask.resize((xm,ym),Image.BICUBIC), (x, y))
        sfondoblack.save('mask.png')


        
    
    
    def ModificasfondoF(): 
        from rembg import remove
        print("modifica sfondo")
        global img,x,y,cx,cy,xm,ym
        
        if os.path.exists(".//canvasbegin.png"):
            img= Image.open(".//canvasbegin.png")
        elif img is None:
            messagebox.showinfo("carica Un immagine")
            return
         
        if img is not None:
            sfondo = Image.new('RGB', (cx, cy), (0, 100, 0))
            sfondowhite = Image.new('RGB', (cx, cy), (255, 255, 255))
            if not os.path.exists(".//canvasbegin.png"):
                xm,ym= img.size    
                if  xm< 1904:
                    x = cx//2 - img.width//2
                else:
                    x=0
                if ym< 1008:
                    y = cy//2 - img.height//2
                else:
                    y=0
            
                
                    
            imageWithoutBg = remove(img)
            # Crea una nuova immagine per la maschera
            mask = Image.new('RGB', imageWithoutBg.size, 'white')

            # Ottieni i dati dell'immagine senza sfondo
            image_data = imageWithoutBg.getdata()

            # Crea una nuova lista per i dati della maschera
            mask_data = []

            # Per ogni pixel nell'immagine senza sfondo
            for item in image_data:
                # Se il pixel è trasparente (ossia, se l'alpha è 0)
                if item[3] == 0:
                    # Aggiungi un pixel bianco alla maschera
                    mask_data.append((255, 255, 255))
                else:
                    # Altrimenti, aggiungi un pixel nero alla maschera
                    mask_data.append((0, 0, 0))

            # Assegna i dati della maschera all'immagine della maschera
            mask.putdata(mask_data)
            # Converte l'immagine della maschera in un array NumPy
            mask_array = np.array(mask)

            # Converte l'array in scala di grigi (necessario per la funzione dilate)
            gray = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)

            # Definisce il kernel per la dilatazione
            kernel = np.ones((5,5),np.uint8)

            # Esegue la dilatazione
            dilation = cv2.dilate(gray,kernel,iterations = 1)
            
            # Converte l'array NumPy in un oggetto Image di PIL
            dilation_image = Image.fromarray(dilation)

            if not os.path.exists(".//canvasbegin.png"):
                # Ora puoi ridimensionare e incollare l'immagine
                sfondowhite.paste(dilation_image.resize((xm,ym),Image.BICUBIC), (x, y))
                sfondo.paste(img.resize((xm,ym),Image.BICUBIC), (x, y))
                sfondo.save('canvasbegin.png')
            else:
                # Ora puoi ridimensionare e incollare l'immagine
                sfondowhite.paste(dilation_image, (0, 0))
                
        

            sfondowhite.save('mask.png')
       
    
    lazobutton= tk.Button(strumenti,text='lazo',bg='white',command= attiva_lazo)
    lazobutton.grid(row=0, column=1,pady=5)

    retangolobutton= tk.Button(strumenti,text='rettangol',bg='white',command= attiva_retangolo)
    retangolobutton.grid(row=1, column=1,pady=5)

    detectvestiti= tk.Button(strumenti,text="rileva vestiti",bg= "white",command= rilevavestitiF)
    detectvestiti.grid(row=0,column=2,pady=5)
        
    modificasf= tk.Button(strumenti,text="Modifica sfondo",bg= "white",command= ModificasfondoF)
    modificasf.grid(row=1,column=2,pady=5)
    
    resetmask= tk.Button(strumenti,text="reset mask",command= delmask)
    resetmask.grid(row=0,column=3,pady=5)
    
    buttonfillgeneration= tk.Button(strumenti,text= "riem_generativo", command= fillgenerate)
    buttonfillgeneration.grid(row=2, column=0, pady=10)

    riniscifaccia= ttk.Combobox(strumenti,values=['rifinisci faccia','non rifinisci faccia'])
    riniscifaccia.grid(row=2, column=1, pady=2)
    riniscifaccia.set('non rifinisci faccia')

    riniscifacciabutton= tk.Button(strumenti,text='rifinisci faccia',command=rifinisciF)
    riniscifacciabutton.grid(row=2, column=2, pady=2)

    FaceActor= tk.Button(strumenti,text='Face Actor',command=FaceActorf)
    FaceActor.grid(row=2, column=3, pady=2)
    

    labtesto= tk.Label(strumenti,text='prompt positive')
    labtesto.grid(row=3, column=0)

    labnegativo= tk.Label(strumenti,text='prompt negative')
    labnegativo.grid(row=3, column=1)

    testo= tk.Text(strumenti,width=20,height=10)
    testo.grid(row=4, column=0)

    negative= tk.Text(strumenti,width=20,height=10)
    negative.grid(row=4, column=1)
    
     
    # bottone mostra immagine
    def mostracaricaF():
        global imageface
        print("mostra o carica una faccia per il faceActor")
        if imageface is None:
            print("Nessuna immagine faccia")
            imaF = cv2.imread('.//I.jpg')
            plt.imshow(cv2.cvtColor(imaF, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            imaF = cv2.imread(imageface)
            plt.imshow(cv2.cvtColor(imaF, cv2.COLOR_BGR2RGB))
            plt.show()

        filename = filedialog.askopenfilename(filetypes=[('Image Files', '*.jpg;*.png')])
        if filename:
            # mostra immagine selezionata
            imageface = filename
            imaF = cv2.imread(imageface)
            plt.imshow(cv2.cvtColor(imaF, cv2.COLOR_BGR2RGB))
            plt.show()
    buttonmostraface= tk.Button(strumenti,text="Mostra/Carica Faceactor", command= mostracaricaF)
    buttonmostraface.grid(row=4,column=2)
        
    labmodel= tk.Label(strumenti,text="modello")
    labmodel.grid(row= 5,column=0)
    
    labcfg= tk.Label(strumenti,text="CFG")
    labcfg.grid(row=5, column=1)

    labsteps= tk.Label(strumenti,text="Steps")
    labsteps.grid(row=5, column=2)
    
    combomodel= ttk.Combobox(strumenti,values=['Inpaint_V6','Inpaint_SDXL1.0','stablediffusion','NUDE','NUDE2','VULVA','NUDE_Refine'])
    combomodel.grid(row=6, column=0)
    combomodel.set('Inpaint_V6')
    
    cfg= tk.Entry(strumenti)
    cfg.grid(row=6, column=1)
    cfg.insert(0, '7.5')

    steps= tk.Entry(strumenti)
    steps.grid(row=6, column=2)
    steps.insert(0, '50')
    
    combobox_reduce = ttk.Combobox(strumenti, values=['riduci immagine', 'non riduci immagine'])
    combobox_reduce.grid(row=7, column=0)
    combobox_reduce.set('riduci immagine')  # Imposta il valore predefinito
    
    labupscale= tk.Label(strumenti,text='UPSCALE')
    labupscale.grid(row=8,column=0)
    comboupscale = ttk.Combobox(strumenti, values=['fast upscale','X2','X3','X4','X8'] )
    comboupscale.grid(row=9,column=0)
    comboupscale.set('fast upscale')
    
    def Upscale():
        global comboupscale, img, canvas, imageface
        print("Upscale")
        # Upscale
        if os.path.exists(".//canvas.png"):
            Imaginecanv = Image.open(".//canvas.png")
        else:
            if imageface is None:
                messagebox.showinfo("Nessuna immagine")
                return
            else:
                Imaginecanv = Image.open(imageface)
        xmi, ymi = Imaginecanv.size
        if comboupscale.get() == 'fast upscale':
            upscalfast = Imaginecanv.resize((xmi*2, ymi*2), Image.BICUBIC)
            upscalfast.save('canvasUP.png')
            img = upscalfast
            photoImg = ImageTk.PhotoImage(upscalfast)  # Salva l'oggetto PhotoImage in una variabile globale
            canvas.create_image(0, 0, image=photoImg, anchor='nw')
            canvas.update()
        elif comboupscale.get() == "X2":
            if xmi> 720 or ymi> 720:
                Imaginecanv.thumbnail((720,720), Image.BICUBIC)
                Imaginecanv = Imaginecanv.convert("RGB")
            model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=2).to("cuda").half()
            inputs = ImageLoader.load_image(Imaginecanv).to("cuda").half()  # Convert inputs to Half before inference
            preds = model(inputs)
            preds = preds.float()  # Convert to float before saving
            ImageLoader.save_image(preds, 'canvasUP.png')
            Imaginecanv = Image.open(".//canvasUP.png")
            img = Imaginecanv
            photoImg = ImageTk.PhotoImage(Imaginecanv)  # Salva l'oggetto PhotoImage in una variabile globale
            canvas.create_image(0, 0, image=photoImg, anchor='nw')
            canvas.update()
        elif comboupscale.get() == "X3":
            if xmi> 720 or ymi> 720:
                Imaginecanv.thumbnail((720,720), Image.BICUBIC)
                Imaginecanv = Imaginecanv.convert("RGB")
            model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=3).to("cuda").half()
            inputs = ImageLoader.load_image(Imaginecanv).to("cuda").half()  # Convert inputs to Half before inference
            preds = model(inputs)
            preds = preds.float()  # Convert to float before saving
            ImageLoader.save_image(preds, 'canvasUP.png')
            Imaginecanv = Image.open(".//canvasUP.png")
            img = Imaginecanv
            photoImg = ImageTk.PhotoImage(Imaginecanv)  # Salva l'oggetto PhotoImage in una variabile globale
            canvas.create_image(0, 0, image=photoImg, anchor='nw')
            canvas.update()
        elif comboupscale.get() == "X4":
            if xmi> 720 or ymi> 720:
                Imaginecanv.thumbnail((720,720), Image.BICUBIC)
                Imaginecanv = Imaginecanv.convert("RGB")
            model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=4).to("cuda").half()
            inputs = ImageLoader.load_image(Imaginecanv).to("cuda").half()  # Convert inputs to Half before inference
            preds = model(inputs)
            preds = preds.float()  # Convert to float before saving
            ImageLoader.save_image(preds, 'canvasUP.png')
            Imaginecanv = Image.open(".//canvasUP.png")
            img = Imaginecanv
            photoImg = ImageTk.PhotoImage(Imaginecanv)  # Salva l'oggetto PhotoImage in una variabile globale
            canvas.create_image(0, 0, image=photoImg, anchor='nw')
            canvas.update()
        elif comboupscale.get() == "X8":
            if xmi > 720 or ymi > 720:
                Imaginecanv.thumbnail((720,720), Image.BICUBIC)
                Imaginecanv = Imaginecanv.convert("RGB")
        model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=4).to("cuda").half()
        inputs = ImageLoader.load_image(Imaginecanv).to("cuda").half()  # Convert inputs to Half before inference
        preds = model(inputs)
        preds = preds.float()  # Convert to float before saving
        ImageLoader.save_image(preds, 'canvasUP.png')
        Imaginecanv = Image.open(".//canvasUP.png")
        xup, yup = Imaginecanv.size
        upscalex8 = Imaginecanv.resize((xup*2, yup*2), Image.BICUBIC)
        img = upscalex8
        upscalex8.save(".//canvas.png")
        photoImg = ImageTk.PhotoImage(upscalex8)  # Salva l'oggetto PhotoImage in una variabile globale
        canvas.create_image(0, 0, image=photoImg, anchor='nw')
        canvas.update()
                
                
        
    
    buttonUpscale= tk.Button(strumenti, text='Upscale',command= Upscale)
    buttonUpscale.grid(row=10, column=0)
    
    def enhancerF():
        global combobackgraund,combomodelenha,comboupscale,imagepath
        print(imagepath)
        
        if os.path.exists(".//canvas.png"):
            shutil.copyfile(".//canvas.png",".//photo-enhancer//samples//canvas.png")
        else:
            shutil.copyfile(imagepath,".//photo-enhancer//samples//canvas.png")
            
        if comboupscale.get() in ["X2","X4"]:
            print("Upscale enhancer")
            os.chdir("photo-enhancer")
            if combobackgraund.get()== "background_enha":
                os.system(f"python main.py --method {combomodelenha.get()} --image_path samples/canvas.png --output_path ..//enha_{combomodelenha.get()}_{comboupscale.get()}.jpg --background_enhancement --upscale {int(comboupscale.get().replace('X',''))}")
            elif combobackgraund.get()== "No background_enha":
                os.system(f"python main.py --method {combomodelenha.get()} --image_path samples/canvas.png --output_path ..//enha_{combomodelenha.get()}_{comboupscale.get()}.jpg --upscale {int(comboupscale.get().replace('X',''))}")
            os.chdir("..")
            while not os.path.exists(f".//enha_{combomodelenha.get()}_{comboupscale.get()}.jpg"):
                print ("attendi")
            imgeh= Image.open(f"enha_{combomodelenha.get()}_{comboupscale.get()}.jpg")
            plt.imshow(imgeh)
            plt.waitforbuttonpress()       
                
        else:
            messagebox.showinfo("puo upscalare enhancer solo X2, X4")

        
    combobackgraund= ttk.Combobox(strumenti,values=['background_enha','No background_enha'])
    combobackgraund.grid(row=9,column=1)
    combobackgraund.set('background_enha')
    
    combomodelenha= ttk.Combobox(strumenti,values=['gfpgan','RestoreFormer'])
    combomodelenha.grid(row=9,column=2)
    combomodelenha.set('gfpgan')
    
    button_enhancer= tk.Button(strumenti,text="enhancer",command=enhancerF)
    button_enhancer.grid(row=10,column=1)
    
    def mostraM():
        print("mostra Maschera")
        if os.path.exists(".\\mask.png"):
            # se il plottere showmask è gia aperto chiudilo e riaprilo con nuova mask
            if plt.fignum_exists(1):
                plt.close(1)  
            maskM = cv2.imread('.\\mask.png')
            plt.figure(1)
            plt.imshow(cv2.cvtColor(maskM, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            messagebox.showinfo("Avviso", "Non c'è nessuna maschera da mostrare")
            
    buttonmaskmostra= tk.Button(strumenti, text='Mostra Maschera',command= mostraM)
    buttonmaskmostra.grid(row=11,column=1,pady=20)
    
    def ridimansionaImage(val):
        global RidImg,canvas,photoImg,imagepath,Img,positionImgX,positionImgY,xm,ym
        if not os.path.exists('.\\canvas.png') and imagepath is None:
            print("nessun Immagine")
        elif imagepath is not None: 
            Img = Image.open(imagepath)
        if os.path.exists('.\\canvas.png') or not imagepath is None or not Img is None:     
            xm,ym= Img.size
            aspect_ratio = xm / ym
            if xm>= ym:
                xm= int(RidImg.get())
                ym= int(xm / aspect_ratio)
            else:
                ym= int(RidImg.get())
                xm= int(ym * aspect_ratio)
                    
            Img= Img.resize((xm,ym),Image.BICUBIC)
            canvas.delete('all')
            canvas.config(bg="dark green")
            photoImg = ImageTk.PhotoImage(Img)  # Salva l'oggetto PhotoImage in una variabile globale
            canvas.create_image(positionImgX.get(), positionImgY.get(), image=photoImg, anchor='nw')
            canvas.update()
            if os.path.exists(".\\canvas.png"):
                os.remove(".\\canvas.png")
            sf=Image.new('RGB', (1904, 1008), (0, 100, 0))  # Utilizza il codice colore RGB per il verde scuro
            sf.paste(Img,(positionImgX.get(),positionImgY.get()))
            sf.save(".\\canvasbegin.png")
    def posiz_image(val):
        global positionImgY,positionImgX,canvas,photoImg,imagepath ,Img
        if Img is None:
            if not os.path.exists('.\\canvas.png') and imagepath is None:
                print("nessun Immagine")
            elif os.path.exists('.\\canvas.png'):
                Img = Image.open(".\\canvas.png")
            elif imagepath is not None: 
                Img = Image.open(imagepath)
                
        if os.path.exists('.\\canvas.png') or not imagepath is  None or not Img is None:
                canvas.delete('all')
                canvas.config(bg="dark green")
                photoImg = ImageTk.PhotoImage(Img)  # Salva l'oggetto PhotoImage in una variabile globale
                canvas.create_image(positionImgX.get(), positionImgY.get(), image=photoImg, anchor='nw')
                canvas.update()
                if os.path.exists(".\\canvas.png"):
                   os.remove(".\\canvas.png")
                sf=Image.new('RGB', (1904, 1008), (0, 100, 0))  # Utilizza il codice colore RGB per il verde scuro
                sf.paste(Img,(positionImgX.get(),positionImgY.get()))
                sf.save(".\\canvasbegin.png")

    #ridimensiona Imagine
    RidImg = Scale(strumenti, from_=500, to=2000, orient=HORIZONTAL,command= ridimansionaImage)
    # Inizializza trackbar con dimansione maggiore imagine
    if xm>= ym:
        RidImg.set(value=xm)
    else:
        RidImg.set(value=ym)

    RidImg.grid(row=12,column=0)
    
    positionImgX= Scale(strumenti, from_=0, to_= 1904,orient=HORIZONTAL,command= posiz_image)
    positionImgX.set(value=0)
    positionImgX.grid(row=13,column=0)

    positionImgY= Scale(strumenti, from_= 0, to_= 1008 ,orient=HORIZONTAL,command= posiz_image)
    positionImgY.set(value=0)
    positionImgY.grid(row=14,column=0)
    def centerXF():
        global positionImgX,img
        if img== None:
            positionImgX.set(int(1904/2))
        else:
            xm,ym= img.size
            positionImgX.set(int((1904-xm)/2))
            
    def centerYF():
        global positionImgY,img
        if img== None:
            positionImgY.set(int(1904/2))
        else:
            xm,ym= img.size
            positionImgY.set(int((1008-ym)/2))
            
    centerx= tk.Button(strumenti,text="centra X", command= centerXF)
    centerx.grid(row=13 ,column=1)
    centery= tk.Button(strumenti,text="centra Y", command= centerYF)
    centery.grid(row=14 ,column=1)
    
    # Creazione e posizionamento dei widget
    labelstreng = tk.Label(strumenti, text="strength")
    labelstreng.grid(row=15, column=0)
    trackstrength = Scale(strumenti, from_=0.0, to_=1.0, orient=HORIZONTAL, resolution=0.01)
    trackstrength.grid(row=16, column=0)
    trackstrength.set(0.55)

    labelseta = tk.Label(strumenti, text="eta")
    labelseta.grid(row=15, column=1)
    tracketa = Scale(strumenti, from_=0.0, to_=1.0, orient=HORIZONTAL, resolution=0.01)
    tracketa.grid(row=16, column=1)
    tracketa.set(0.55)
    def estF():
        print("estrai frame")
        os.system("python estraiframe.py")
    buttonestraframe= tk.Button(strumenti,text="Estrai frame Video",command= estF)
    buttonestraframe.grid(row=17, column=0,pady=10)
          
    lablora= tk.Label(strumenti,text="Lora Model_:")
    lablora.grid(row=18,column=0)
    lorarefine = ttk.Combobox(strumenti, values=['No_lora','Phoebe','Piper','Prue','Paige','Billie','perfection_style_SD1.5','perfection_style','p*ss*'])
    lorarefine.grid(row=18, column=1)
    lorarefine.set('perfection_style_SD1.5')
        
    strumenti.mainloop()
    
    



q = queue.Queue()
Tstrumenti = threading.Thread(target=create_ui, args=(q,))
Tstrumenti.start()
def openfile():
    global photoImg, w, h, img,points,cx,cy,salvasfondo,imageface,imagepath,positionImgX,positionImgY,x,y
    salvasfondo= False
    points=[]
    if os.path.exists(".//canvasbegin.png"):
        os.remove(".//canvasbegin.png")
    if os.path.exists(".//canvas.png"):
            os.remove(".//canvas.png")
    if os.path.exists(".//mask.png"):
        os.remove(".//mask.png")
    if os.path.exists(".//oldmask.png"):
            os.remove(".//oldmask.png")
    if os.path.exists(".//mask1.png"):
        os.remove(".//mask1.png")
    
     
        
    imagepath = filedialog.askopenfilename(title="scegli un immagine", filetypes=(("Image files", "*.jpg *.png"), ("All files", "*.*")))
    if imagepath:
        # Open the image
        img =  Image.open(imagepath)
        imageface= imagepath
        w, h = img.size

        # Calculate the ratio of the new width to the old width
        ratio = min(cx/w, cy/h)
        w = int(w * ratio)
        h = int(h * ratio)

        # Resize the image
        img = img.resize((w, h), Image.LANCZOS)

        # Now, create the PhotoImage object
        photoImg = ImageTk.PhotoImage(img)

        # Center the image
        x = (cx - w) // 2
        y = (cy - h) // 2
        canvas.create_image(x,y, image=photoImg, anchor='nw')
        
       

 
    

def lazo1(event):
    global points, last_point, SELECTLAZO, SELECTRETANGOLO
    if SELECTLAZO == True:
        points.append((event.x, event.y))
        # Draw a line from the last point to the current point
        if last_point is not None:
            canvas.create_line(last_point, points[-1], fill="yellow")  # Change the color to yellow
        last_point = points[-1]
    if SELECTRETANGOLO == True:
        pointsrettangle.append((event.x, event.y))
        if len(pointsrettangle) > 1:
            canvas.delete("rectangle")
            canvas.create_rectangle(pointsrettangle[0][0], pointsrettangle[0][1], pointsrettangle[-1][0], pointsrettangle[-1][1], outline="yellow", tags="rectangle") 
        last_point = pointsrettangle[-1]

def save_canvas():
    global img,cx,cy,salvasfondo,positionImgX,positionImgY,xm,ym,x,y
    # Crea una nuova immagine con sfondo verde
    sfondo = Image.new('RGB', (cx, cy), (0, 100, 0))  # Utilizza il codice colore RGB per il verde scuro
    # Calcola le coordinate per centrare l'immagine
    
    
     
    xm,ym= img.size    
    if  xm< 1904:
        # Centra l'immagine se le coordinate non sono valide
        x = cx//2 - img.width//2
    else:
        x=0
    if ym< 1008:
        y = cy//2 - img.height//2
    else:
        y=0
    # Sovrapponi l'immagine allo sfondo
    sfondo.paste(img.resize((xm,ym),Image.BICUBIC), (x, y))
    # Salva l'immagine risultante
    sfondo.save('canvasbegin.png')
    salvasfondo= True


def lazo2(event):
    global points, last_point,SELECTLAZO,SELECTRETANGOLO,pointsrettangle,canvas 
    if SELECTLAZO== True:
        # Add the release point to the points list
        points.append((event.x, event.y))
        # Check if there is a point in the points array that is close to the start point
        for point in points:
            if abs(points[0][0] - point[0]) < 10 and abs(points[0][1] - point[1]) < 10:
                print("mask")
                creamaskera()
                break
        
    if SELECTRETANGOLO == True:
        print("mask rettangolo")
        creamaskera()
        pointsrettangle.clear()  # Clear the list when releasing the mouse button
         

    

def creamaskera():
    global points,img,w,h,SELECTLAZO,SELECTRETANGOLO,pointsrettangle,cx,cy,last_point,salvasfondo,ragioblur
    
    if SELECTLAZO==True:
        if len(points) < 2:
            print("Not enough points to create mask")
            return
        if not os.path.exists('.\\canvasbegin.png'):
           save_canvas()
        canvas_img = Image.open('canvasbegin.png')
        canvasw,canvash = canvas_img.size  # Ottieni le dimensioni dell'immagine del canvas
        canvasw,canvash = canvas_img.size  # Ottieni le dimensioni dell'immagine del canvas
        # Create a new image with the same size as the canvas
        mask = Image.new('L', (cx, cy), 0)
        # Calculate the image offset
        x_offset = (cx - canvasw) // 2
        y_offset = (cy - canvash) // 2
        # Adjust the points for the offset
        adjusted_points = [(x-x_offset, y-y_offset) for x, y in points]
        # Draw the polygon (lazo selection) on the mask image
        ImageDraw.Draw(mask).polygon(adjusted_points, outline=255, fill=255)
        # Save the mask image
        if os.path.exists('mask.png')== False:
            mask.save('mask.png')
            points=[]
            last_point = None
        else:
            mask.save('mask1.png')
            mask1= Image.open('mask.png')
            mask1.save('oldmask.png')
            mask1.paste(mask,(0,0),mask)
            mask1.save('mask.png')
            # Remove the temporary mask
            os.remove("mask1.png")
            points=[]
            last_point = None
        clear_canvas()  # Clear the canvas after restoring the mask
        
    elif SELECTRETANGOLO == True:
        if len(pointsrettangle) < 2:
            print("Not enough points to create mask")
            return
        
        if not os.path.exists('.\\canvasbegin.png'):
           save_canvas()
        canvas_img = Image.open('canvasbegin.png')
        canvasw, canvash = canvas_img.size  # Ottieni le dimensioni dell'immagine del canvas
        # Create a new image with the same size as the canvas
        mask = Image.new('L', (cx, cy), 0)
        # Calculate the image offset
        x_offset = (cx - canvasw) // 2
        y_offset = (cy - canvash) // 2
        # Adjust the points for the offset
        adjusted_points = [(x-x_offset, y-y_offset) for x, y in pointsrettangle]
        # Sort the points before drawing the rectangle
        upper_left_point = min(adjusted_points[0][0], adjusted_points[-1][0]), min(adjusted_points[0][1], adjusted_points[-1][1])
        lower_right_point = max(adjusted_points[0][0], adjusted_points[-1][0]), max(adjusted_points[0][1], adjusted_points[-1][1])
        # Draw the rectangle on the mask image
        ImageDraw.Draw(mask).rectangle([upper_left_point, lower_right_point], outline=255, fill=255)
        # Save the mask image
        if os.path.exists('mask.png') == False:
            mask.save('mask.png')
            pointsrettangle = []
        else:
            mask.save('mask1.png')
            mask1 = Image.open('mask.png')
            mask1.save('oldmask.png')
            mask1.paste(mask, (0,0), mask)
            mask1.save('mask.png')
            # Remove the temporary mask
            os.remove("mask1.png")
            pointsrettangle = []
        clear_canvas()  # Clear the canvas after restoring the mask
     

def clear_canvas():
    global canvas,photoImg
    canvas.delete("all")
    if os.path.exists('.//canvas.png'):
        canvas_img = Image.open('canvas.png')
    else:
        canvas_img = Image.open('canvasbegin.png')
    # Now, create the PhotoImage object
    photoImg = ImageTk.PhotoImage(canvas_img)
    canvas.create_image(0, 0, image=photoImg, anchor='nw')
    
def indietro(event):
    print("indietro")
    if os.path.exists('.//mask.png'):
        os.remove('.//mask.png')
        if os.path.exists('.//oldmask.png'):
            os.rename('.//oldmask.png', './/mask.png')  
    clear_canvas()  # Clear the canvas after restoring the mask
        
canvas.bind("<Button-1>", lazo1)
canvas.bind("<Button-3>", indietro)
canvas.bind("<B1-Motion>", lazo1)  # Bind to mouse motion event with button 1 down
canvas.bind("<ButtonRelease-1>", lazo2)

#se chiudo il form di windows chiudi anche il form degli strumenti
#def on_closing():
#   os.system("taskkill /f /im python.exe")
#window.protocol("WM_DELETE_WINDOW", on_closing)

window.mainloop()
