from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width , get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # Carregar o modelo YOLO pre treinado
        self.tracker = sv.ByteTrack()  # Inicializar o rastreador


    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position


    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate as posições da bola  e preencher os valores faltantes
        df_ball_positions = df_ball_positions.interpolate() # preencher os valores faltantes em geral
        df_ball_positions = df_ball_positions.bfill() # preencher os valores faltantes no início do vídeo

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
            batch_size=20 # Definir o tamanho do lote para processamento em lotes
            detections = [] 
            for i in range(0,len(frames),batch_size):
                detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)# Realizar a detecção em lotes de frames
                detections += detections_batch
            return detections# Adicionar as detecções do lote à lista geral de detecções

   
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
                #o stub é um arquivo json que contém as detecções e os rastreamentos dos objetos, caso queira ler de um arquivo em vez de processar o vídeo novamente

        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f) #ler as detecções e os rastreamentos dos objetos de um arquivo pickle 
            return tracks
# caso ja exista um arquivo com as detecções ,ele vai ler esse aquivo ao invez de rodar tudo de novo
        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()} # Criar um dicionário invertido para mapear os índices de classe para os nomes das classes

            #converter as detecçoes para o Supervision
            detection_supervision = sv.Detections.from_ultralytics(detection) 

            # converter o goleiro para jogador 
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper": # Verificar se a classe é "goleiro", "goalkeeper" como é indentificado no modelo
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"] # Substituir a classe "goleiro" por "jogador", "player" como é indentificado no modelo

            #trackear os objetos
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision) #vai adicionar um traker na detecção

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox} 
                    # Adicionar a caixa delimitadora ao dicionário de rastreamento do jogador usando o track_id como chave

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox} 
                    # Adicionar a caixa delimitadora ao dicionário de rastreamento do árbitro usando o track_id como chave

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()# Obter as coordenadas da caixa delimitadora
                cls_id = frame_detection[3] 
                
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f) #salvar as detecções e os rastreamentos dos objetos em um arquivo pickle


        return tracks #uma lista de dicionarios
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3]) # y2 é a coordenado do pe do jogador, que é onde a elipse deve ser desenhada
        x_center, _ = get_center_of_bbox(bbox) # so usamos o cento x pois o y ja foi pego
        width = get_bbox_width(bbox)

        cv2.ellipse(   #desenhando a elipse
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)), #fornece o raio da elipse 
            angle=0.0, 
            startAngle=-45,  #esse
            endAngle=235,    # 2 valores não são desenhados para dar um efeito de elipse aberta
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # o trak ID , desenhamos um retanglo e colocamos ele na posição correta    
        rectangle_width = 40
        rectangle_height=20  
        x1_rect = x_center - rectangle_width//2  # x e o centro da caixa 
        x2_rect = x_center + rectangle_width//2  
        y1_rect = (y2- rectangle_height//2) +15  #y é a cordenada do pe
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED) #garante o retangulo preenchido com a cor do time
            
            x1_text = x1_rect+12 #posiciona o texto no meio do retangulo
            if track_id > 99: #caso seja um numero maior , ajustamos a posição
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)), #onde o texto vai se colocado
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0), # cor do numero preto 
                2
            )        

        return frame
    

    def draw_traingle(self,frame,bbox,color): 
        y= int(bbox[1]) # a parte de baixo do bbox , e vai ficar no topo da bola
        x,_ = get_center_of_bbox(bbox) 

        triangle_points = np.array([  
            [x,y], #primeiro ponto é o topo da bola
            [x-10,y-20], # esses 2 pontos formam a base do trangulo
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED) #desenha o triangulo preenchido com a cor fornecida
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2) # desenha uma borda preta no triangulo 

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # um retangulo transparente  
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 ) #posição , cor , cheio
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # pega a porcentagem de tempo que o time ficou com a bola
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0] 
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

#como ira aparecer no retangulo 
        cv2.putText(frame, f"time 1 Controle de bola: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Time 2 Controle de bola: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
    




    def draw_annotations(self,video_frames, tracks,team_ball_control): # desenha de maneira personalisada as anotações nos frames do vídeo
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # desenhar os jogadores
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)#desenha a elipse e o track_id do jogador

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

             # desenhar o árbitro usando a cor amarela
            for _, referee in referee_dict.items(): # as estatisticas do arbrito não são tão importantes 
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))

            # desenha a bola , cor
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))             

            # controle de bola 
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames