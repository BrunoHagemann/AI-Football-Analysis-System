from ultralytics import YOLO 

model = YOLO('models/best.pt') # a diferença de verção soão os params 

results = model.predict(source='input_videos/08fd33_4.mp4', save=True)
print(results[0])
print("================================")
for box in results[0].boxes:
    print(box)  # Coordenadas da caixa delimitadora
   