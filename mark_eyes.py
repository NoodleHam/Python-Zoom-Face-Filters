from glob import glob
import threading
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# written in python3


def prompt_eye_locations(result, img_paths):
    assert isinstance(result, list)
    image_index = int(input("Pick your filter index (int): "))
    loc_left_eye = input(
        "Type in the rounded integer x, y-coordinate for left eye, separated with a comma\n")
    loc_left_eye = list(map(int, loc_left_eye.split(",")))
    result.append(loc_left_eye)
    print(loc_left_eye)
    loc_right_eye = input(
        "Type in the rounded integer x,y-coordinate for the right eye, separated with a comma\n")
    loc_right_eye = list(map(int, loc_right_eye.split(",")))
    result.append(loc_right_eye)
    print(loc_right_eye)
    image_name = str(img_paths[image_index])
    image_name = image_name[: image_name.rfind(".")]
    json_name = image_name + "_annotation.json"
    json_dict = {"left_eye_x_y":loc_left_eye, "right_eye_x_y":loc_right_eye}
    save = open(json_name, "w")
    json.dump(json_dict, save)
    save.close()
    print("file saved! close all windows to continue.")


images = glob("filter_imgs/*.png") + glob("filter_imgs/*.jpg")
for i, img_file in enumerate(images):
    slash_index = max(img_file.rfind("/"), img_file.rfind("\\")) + 1
    img_name = img_file[slash_index:]
    print(img_name)
    img = mpimg.imread(img_file)
    f = plt.figure()
    title = "Image {}: {}".format(i, img_name)
    f.canvas.set_window_title(title)
    plt.title("")
    plt.imshow(img)

plt.draw()
res = []
# t = threading.Thread(target=prompt_eye_locations, args=(res, images))
# t.setDaemon(True)
# t.start()
plt.show()
# t.join()





