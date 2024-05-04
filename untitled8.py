import streamlit as st
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Définir le chemin vers le répertoire contenant les images
data_dir = r"C:\Users\HP\Desktop\hiba ml\newdata\dataset"

# Définir les dimensions de l'entrée et le nombre de classes (émotions)
input_shape = (48, 48, 1)  # Taille de l'image en niveaux de gris
num_classes = 7  # Nombre d'émotions différentes

# Fonction pour créer et compiler le modèle CNN
def create_emotion_detection_model():
    model = Sequential()
    
    # Première couche de convolution
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Deuxième couche de convolution
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Troisième couche de convolution
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Couche de mise en forme
    model.add(Flatten())
    
    # Couche dense
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    # Couche de sortie avec softmax pour la classification des émotions
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compiler le modèle
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Créer le modèle CNN pour la détection d'émotions
emotion_detection_model = create_emotion_detection_model()

# Définir le générateur de données pour l'ensemble d'images
datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_directory(
    data_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical'
)

# Entraîner le modèle CNN sur les données chargées
emotion_detection_model.fit(generator, epochs=20)

# Définir la variable video_capture
video_capture = cv2.VideoCapture(0)

# Définir la fonction pour détecter les émotions
def detect_emotion():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = video_capture.read()
        
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y+h, x:x+w]
                resized_frame = cv2.resize(face_roi, (48, 48))
                normalized_frame = resized_frame / 255.0
                
                input_frame = np.expand_dims(normalized_frame, axis=0)
                input_frame = np.expand_dims(input_frame, axis=-1)
                
                prediction = emotion_detection_model.predict(input_frame)
                emotion_index = np.argmax(prediction)
                emotion_label = ['angry', 'disgust', 'disgusted', 'fear', 'fearful', 'happy', 'neutral', 'sad', 'surprised'][emotion_index]

                # Afficher l'émotion détectée sur la tête de l'utilisateur
                cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
            # Afficher la frame
            cv2.imshow('Video', frame)
            
        else:
            st.write("Erreur lors de la capture de la vidéo.")
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

# Fonction principale pour l'interface Streamlit
def main():

    st.title("Détection d'émotions faciales avec CNN et Streamlit")
    st.sidebar.title("Menu")
    
    menu_options = ["Accueil", "Détecter l'émotion", "À propos"]
    choice = st.sidebar.selectbox("Choix", menu_options)
    
    if choice == "Accueil":
        st.write("## Bienvenue sur notre site de détection d'émotions faciales !")
        st.write("### Voici une image pour vous souhaiter la bienvenue :")
        st.image(r'C:\Users\HP\Desktop\hiba ml\chine-reconnaissance-faciale-carte-sim.jpg', use_column_width=True)
        
    elif choice == "Détecter l'émotion":
        st.write("Cliquez sur le bouton ci-dessous pour ouvrir la caméra et détecter l'émotion.")
        if st.button("Ouvrir la caméra"):
            detect_emotion()
        
    elif choice == "À propos":
        st.write("### À propos de ce projet...")
        st.write("Ceci est un projet de détection d'émotions faciales utilisant un modèle CNN.")
        st.write("Le modèle est entraîné pour détecter les émotions suivantes : colère, dégoût, peur, joie, tristesse, surprise et neutre.")
        st.write("Ce projet a été réalisé par des étudiantes de ENSATE")
        
        
# Appeler la fonction principale
if __name__ == "__main__":
    main()
