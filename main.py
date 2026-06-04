from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    #ler video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #inicializar o rastreador
    tracker = Tracker('models/best.pt')

    #obter as detecções e os rastreamentos dos objetos
    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub= True,
                                        stub_path='stubs/track_stubs.pkl') 
    #definir o caminho do stub para ler as detecções e os rastreamentos dos objetos de um arquivo picklee

    # pega a posição dos objetos 
    tracker.add_position_to_tracks(tracks)


    # movimento de camera
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

        #  o View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    #interpolate as poisçoes da bola 
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # estimativa de velocidade e distancia 
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)


    #jogadores dos times
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']): # loop para cada frame e cada jogador
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]    

    # quem tem a posse da bola
    player_assigner =PlayerBallAssigner()
    team_ball_control= [] #qual time mantem mais o controle da bola
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1]) # caso tenha um passe de bola , vai contunuar contando o controle da bola considerando o ultimo que tocuo nela
    team_ball_control= np.array(team_ball_control)



        # salva uma pequena imagem do jogador
#    for track_id, player in tracks['players'][0].items():
#        bbox = player['bbox']
#        frame = video_frames[0]

        # pega o corte da imagem usando as coordenadas do bbox
#        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # salva essa imagem 
#        cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)

#        break


    # desenah o output 
    ## desenha o object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks , team_ball_control)

    ##desenhar o movimento da camera
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## desenha a velocidade e a distancia 
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    #salva video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == "__main__":
    main()