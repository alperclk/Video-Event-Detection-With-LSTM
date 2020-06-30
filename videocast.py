#!/usr/bin/env python
# coding: utf-8

# In[47]:


import cv2


# In[48]:


def videoKayit(kayitAdi, frameDizisi, fps, width, height ):
    video_name = 'kayitlar/gol%d.mp4' %kayitAdi
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G',), fps, (width,height))
    
    #frameDizisi=np.reshape(frameDizisi,(-1,224*224*3))
    for i in frameDizisi:
        video.write(i)
        #cv2.imshow('son',i)
        #if cv2.waitKey(25) & 0xFF == ord('q') :
            #break
    
    video.release()
    cv2.destroyAllWindows()


# In[49]:


def videoFrameleriOku(filepath, baslangicFrameNo, bitisFrameNo):
    frameler=[]
    vidcap = cv2.VideoCapture(filepath)    
    count = 0    
    
    while (count < baslangicFrameNo): # kontrol edilecek yere gelene kadar sadece frameleri geç
        success,image = vidcap.read()
        if success :
            count += 1
            #print("gec" + str(count))


    while (count >= baslangicFrameNo and count <= bitisFrameNo ): # oku ve frameleri diziye kaydet
        success,image = vidcap.read()
        if success :
            count += 1            
            frameler.append(image) ###
            #print(count)
    print( str(len(frameler)) + " frame okundu.")
    
    vidcap.release()
    
    
    return frameler


# In[50]:


def hw(filepath):  #height, weight
    vidcap = cv2.VideoCapture(filepath) 
    success,image = vidcap.read()
    if success :
        heightwidth = image.shape[:2]  
        
    vidcap.release()
    print("Height: " + str(heightwidth[0]) + ", Width: " + str(heightwidth[1]) )
    return heightwidth


# In[51]:


def videoFrameSayisi(filepath):
    vidcap = cv2.VideoCapture(filepath) 
    frameSayisi = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    vidcap.release()
    print("Toplam" + str(frameSayisi) + " frame var")
    return frameSayisi


# In[52]:


def fps(filepath):
    vidcap = cv2.VideoCapture(filepath) 
    fpsSayisi = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    print("Fps: " + str(fpsSayisi) )
    return fpsSayisi


# In[57]:


#path="vid/ronaldene.mp4"
path="vid/ilhanm.mp4"
baslangicSaniye= 4
bitisSaniye = 8
kayitAdi=2

fpsS = int(round(fps(path)))
height, width = hw(path)

baslangicFrameNo = baslangicSaniye*fpsS
bitisFrameNo = bitisSaniye*fpsS



videoKayit(kayitAdi, videoFrameleriOku(path, baslangicFrameNo, bitisFrameNo), fpsS, width, height )



# In[56]:



vidcap = cv2.VideoCapture("vid/gol1.mp4") 
frameSayisi = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

vidcap.release()
print("Toplam" + str(frameSayisi) + " frame var")
    

