import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
slide = openslide.OpenSlide('2019-01-21 10.43.12.ndpi')

print (type(properties['hamamatsu.XOffsetFromSlideCentre']))


# associated_images = slide.associated_images
import xml.dom.minidom

annotation = xml.dom.minidom.parse("HE-2.ndpi.ndpa")
root = annotation.documentElement


points = annotation.getElementsByTagName("pointlist")


slide = openslide.OpenSlide('HE-2.ndpi')
properties = slide.properties

level0_dimension = slide.level_dimensions[0]
width, height = float(level0_dimension[0]), float(level0_dimension[1])

x_off = int(properties['hamamatsu.XOffsetFromSlideCentre'])
y_off = int(properties['hamamatsu.YOffsetFromSlideCentre'])
ratio_x = float(properties['openslide.mpp-x']) * 1000
ratio_y = float(properties['openslide.mpp-y']) * 1000

def transfer_x(v):
    return int(width / 2 - x_off / ratio_x + v / ratio_x)

def transfer_y(v):
    return int(height / 2 - y_off / ratio_y + v / ratio_y)

for mmp, i in enumerate(points):
    sigedian = i.getElementsByTagName("point")
    qidian_x = sigedian[0].getElementsByTagName("x")[0].firstChild.data
    qidian_y = sigedian[0].getElementsByTagName("y")[0].firstChild.data
    zhongdian_x = sigedian[2].getElementsByTagName("x")[0].firstChild.data
    zhongdian_y = sigedian[2].getElementsByTagName("y")[0].firstChild.data
    for j in sigedian:
        x = j.getElementsByTagName("x")[0].firstChild.data
        y = j.getElementsByTagName("y")[0].firstChild.data

    x_b = int(zhongdian_x) - int(qidian_x)
    y_b = int(zhongdian_y) - int(qidian_y)

    a_x = transfer_x(int(qidian_x))

    a_y = transfer_y(int(qidian_y))

    print (a_x,a_y)
    im = slide.read_region((a_x,a_y),0,(int(x_b/ratio_x),int(y_b/ratio_y))).convert('RGB')
    im.show()

slide.close()