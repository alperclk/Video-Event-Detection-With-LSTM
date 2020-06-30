#!/usr/bin/env python
# coding: utf-8

# In[35]:


from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Activation,Flatten,Dense,Dropout
import numpy as np
import cv2


data_dim = 63
timesteps = 224*224*3
num_classes = 2

model = Sequential()
model.add(LSTM(64, return_sequences=True,
               input_shape=(timesteps, data_dim))) 
model.add(Dropout(0.2))
model.add(LSTM(63, return_sequences=False)) 
model.add(Dropout(0.2))

model.add(Dense(63, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.load_weights("lstmModel")


# In[36]:


def videoFrameSayisi(filepath):
    vidcap = cv2.VideoCapture(filepath) 
    frameSayisi = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    vidcap.release()
    return frameSayisi


# In[37]:


def fps(filepath):
    vidcap = cv2.VideoCapture(filepath) 
    fpsSayisi = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return fpsSayisi


# In[38]:


def hw(filepath):  #height, weight
    vidcap = cv2.VideoCapture(filepath) 
    success,image = vidcap.read()
    if success :
        heightwidth = image.shape[:2]  
        
    vidcap.release()
    return heightwidth


# In[39]:


def vidoku(filepath, kontrolBaslangicFrameNo):
    imarray=np.array([])
    vidcap = cv2.VideoCapture(filepath)    
    count = 0
    
    
    while (count < kontrolBaslangicFrameNo): # kontrol edilecek yere gelene kadar sadece frameleri geç
        success,image = vidcap.read()
        if success :
            count += 1
            #print("gec" + str(count))


    while (count >= kontrolBaslangicFrameNo and count < kontrolBaslangicFrameNo + 63): # oku ve frameleri diziye kaydet
        success,image = vidcap.read()
        if success :
            count += 1
            image=cv2.resize(image,(224,224))
            imarray=np.append(imarray,image)
            #print(count)
    print( str(count) + " okundu.")
    
    vidcap.release()
    
    return imarray
            


# In[40]:


def videoFrameleriOku(filepath, kontrolBaslangicFrameNo):
    frameler=[]
    vidcap = cv2.VideoCapture(filepath)    
    count = 0    
    
    while (count < kontrolBaslangicFrameNo): # kontrol edilecek yere gelene kadar sadece frameleri geç
        success,image = vidcap.read()
        if success :
            count += 1
            #print("gec" + str(count))


    while (count >= kontrolBaslangicFrameNo and count < kontrolBaslangicFrameNo + 63): # oku ve frameleri diziye kaydet
        success,image = vidcap.read()
        if success :
            count += 1            
            frameler.append(image) ###
            #print(count)
    print( str(count) + " gercek foto okundu.")
    
    vidcap.release()
    
    
    return frameler


# In[41]:


def diziIslemleri(tahmindizisi):
    tahmindizisi=np.reshape(tahmindizisi,(-1,224*224*3))
    tahmindizisi=np.reshape(tahmindizisi,(1,224*224*3,63))
    print(tahmindizisi.shape)  
    return tahmindizisi


# In[42]:


def anliktahmin(predict_array):
    sonuc=model.predict(predict_array)
    print(sonuc)
    return sonuc


# In[43]:


def videoKayit(kayitAdi, frameDizisi, fps, width, height ):
    video_name = 'records/gol%d.mp4' %kayitAdi
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G',), fps, (width,height))
    
    c = 0
    #frameDizisi=np.reshape(frameDizisi,(-1,224*224*3))
    for i in frameDizisi:
        video.write(i)
        cv2.imshow('son',i)
        c += 1
        
        if cv2.waitKey(25) & 0xFF == ord('q') :
            break
    print( str(c) + " frame yazıldı")
    video.release()
    cv2.destroyAllWindows()


# In[44]:


def tahmin(filepath):
    frameSayisi = videoFrameSayisi(filepath)
    fpsSayisi = fps(filepath)
    hewi = hw(filepath)
    
    adim = 0
    golBlokSayisi = 0 
    golKayitSayisi = 0 
    kayitSayisi = 0 # kayıt metodu çağrılırken verilecek olan doğrusu
    
    tekGolFrameDizisi = [] # kontrol edilen 63 framelik bloklar peşpeşe gol ise o da tutulacak
    golFrameler = []
    vidFrOku = []
    bosDizi = []
    oncekiKontrol = 0
    
    while ( ( adim * 63 ) + 63 < frameSayisi ):
        print( str( adim * 63 ) + "/" + str( adim * 63 + 63 ) + " arası frameler kontrol için okunuyor..." )
        images=vidoku(path, (adim * 63 ) )   
        sonuc = anliktahmin(diziIslemleri(images))    
        
        gol=sonuc[0][0]
        faul=sonuc[0][1]
        if(gol>faul):  # gol ise
            print("GOL BLOĞU TESPİT EDİLDİ")
            golBlokSayisi += 1            
            
            if ( ( (adim + 1) * 63 ) + 63 > frameSayisi ): # bir sonra bakacağı blok videonun dışına çıkacaksa (video bitecekse)
                golKayitSayisi += 1
                vidFrOku = videoFrameleriOku(path, (adim * 63 ) )
                print ( len(vidFrOku) ) 
                print("frame var")
                for i in ( (vidFrOku) ):
                    golFrameler.append( i )
                
                #tekGolFrameDizisi = tekGolFrameDizisi.append(images)
                videoKayit ( golKayitSayisi, golFrameler, fpsSayisi, hewi[1], hewi[0]  )  # golü kaydet
                #tekGolFrameDizisi = bosDizi
                #golFrameler = bosDizi
                golFrameler.clear()
                vidFrOku.clear()
            else :    
                
                vidFrOku = videoFrameleriOku(path, (adim * 63 ) )
                print ( len(vidFrOku) ) 
                print("frame var")
                for i in ( (vidFrOku) ):
                    golFrameler.append( i )
                vidFrOku.clear()
                #for i in ( videoFrameleriOku(path, (adim * 63 ) ) ):
                    #golFrameler.append( i )
                    
                #golFrameler.append( videoFrameleriOku(path, (adim * 63 ) ) )
                #tekGolFrameDizisi = tekGolFrameDizisi.append(images)   # gol tespit edileni aklında tutar         
                
            oncekiKontrol = 1    
            adim += 1           
              
            
        else :   # gol değilse
            if ( oncekiKontrol == 1 ): # anlik tespit gol değil ise ancak bir öncesinde gol tespit edilip hafızaya alındıysa
                golKayitSayisi += 1
                videoKayit ( golKayitSayisi, golFrameler, fpsSayisi, hewi[1], hewi[0]  )  # golü kaydet
                #tekGolFrameDizisi = bosDizi
                golFrameler.clear()
            oncekiKontrol = 0
            adim += 1              
            
        
                


# In[ ]:


#path="vid/ronaldene.mp4"
path="test/bilinmeyen.mp4"

tahmin(path)


cv2.destroyAllWindows()


# In[ ]:




