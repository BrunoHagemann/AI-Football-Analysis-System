import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = [] #inicia uma lista para armazenar os frames do vídeo
    
    while True: 
        ret, frame = cap.read()
        if not ret: # analisa se o video continua ou terminou 
            break
        frames.append(frame) #adiciona o frame à lista de frames
    return frames

def save_video(output_video_frames , output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #define o codec de vídeo
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0])) 
    #cria um objeto VideoWriter para salvar o vídeo processado, usando as dimensões do primeiro frame da lista e uma taxa de quadros de 24 fps  
    for frame in output_video_frames: #itera(repete) sobre os frames processados e os escreve no vídeo de saída
        out.write(frame)
    
    out.release() #libera o objeto VideoWriter para finalizar o arquivo de vídeo

