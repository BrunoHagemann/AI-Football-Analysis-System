def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox #calcula o centro do bbox, que é onde a elipse deve ser desenhada
    return int((x1+x2)/2),int((y1+y2)/2) # centro x e centro y do bbox

def get_bbox_width(bbox): # dtetermina o tamnaho da elipse 
    return bbox[2]-bbox[0]

#fazemos uma media das pisiçoes dos pés e direção 
def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1] # para o movimento da camera 

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2) #posição da bolo ao pé 