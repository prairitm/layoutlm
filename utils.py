import json
from shapely.geometry import Polygon
from tqdm import tqdm
from PIL import Image
from lxml import etree
import pandas as pd

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    # print(poly_1,poly_2)
    # iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    iou = poly_1.intersection(poly_2).area
    min_area = min(poly_1.area,poly_2.area)
    return iou/min_area


def hocr_to_dataframe(fp):
    doc = etree.parse(fp)
    print(doc)
    words = []
    wordConf = []
    coords_list = []
    for path in doc.xpath('//*'):
        if 'ocrx_word' in path.values():
            coord_text = path.values()[2].split(';')[0].split(' ')[1:]
            word_coord = list(map(int, coord_text)) #x1, y1, x2, y2
            conf = [x for x in path.values() if 'x_wconf' in x][0]
            wordConf.append(int(conf.split('x_wconf ')[1]))
            words.append(path.text)
            coords_list.append(word_coord)

    dfReturn = pd.DataFrame({'word' : words,
                             'coords': coords_list,
                             'confidence' : wordConf})

    return(dfReturn)