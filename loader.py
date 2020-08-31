import os
import json
import glob
import numpy as np
import cv2 as cv


def data_loader(cg, created_data_path, data_path, json_path):
    fl1, lb1 = created_data_loader(created_data_path, cg)
    fl2, lb2 = json_data_loader(data_path, json_path)
    for i in range(0, len(lb2)):
        lb2[i] = cg.char_to_number(lb2[i])

    fl1.extend(fl2)
    lb1.extend(lb2)
    return fl1, np.array(lb1, dtype=np.int)


def create_data(cg, tg, font_dir, created_data_path):
    font_file_list = glob.glob(os.path.join(font_dir, '*.ttf'))
    font_file_list.extend(glob.glob(os.path.join(font_dir, '*.ttc')))

    size = cg.get_character_count()

    # create test
    for i in range(0, len(font_file_list)):
        basename = os.path.basename(font_file_list[i])
        basename = os.path.splitext(basename)[0]

        font_image_dir = os.path.join(created_data_path, basename)
        if not os.path.exists(font_image_dir):
            os.mkdir(font_image_dir)

        font_size = np.random.randint(low=80, high=150, size=size)
        size_width = np.random.randint(25, 60, size)
        size_height = np.random.randint(25, 60, size)

        for j in range(0, size):
            character = cg.number_to_char(j)

            image = tg.char_to_image(character, font_size[j], i,
                                     (255, 255, 255))
            image = image.resize((size_width[j], size_height[j]))
            image = np.array(image, dtype=np.uint8).reshape(size_height[j], size_width[j], 3)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = cv.bitwise_not(image)
            image[image < 255] = 0
            cv.imwrite(os.path.join(font_image_dir, f"{j}.png"), image)

        print(f"created {font_file_list[i]} image")


def created_data_loader(created_data_path, cg):
    dirs = os.listdir(created_data_path)

    file_names = []
    label_number = []
    count = cg.get_character_count()

    for font_dir in dirs:
        for i in range(0, count):
            file_name = os.path.join(os.path.join(created_data_path, font_dir), f"{i}.png")
            file_names.append(file_name)
            label_number.append(i)

    return file_names, label_number


def json_data_loader(data_path, json_path):
    file_names = []
    label_text = []

    with open(json_path, encoding='UTF8') as json_file:
        json_data = json.load(json_file)

        for i in range(0, 532659):
            file_names.append(os.path.join(data_path, json_data['images'][i]['file_name']))
            label_text.append(json_data['annotations'][i]['text'])

    return file_names, label_text


def hand_written_data_loader(cg, data_path, json_path):
    file_names = []
    label_text = []

    with open(json_path, encoding='UTF8') as json_file:
        json_data = json.load(json_file)
        size = len(json_data['images'])

        for i in range(0, size):
            if not os.path.exists(os.path.join(data_path, json_data['images'][i]['file_name'])):
                continue

            file_names.append(os.path.join(data_path, json_data['images'][i]['file_name']))
            label_text.append(json_data['annotations'][i]['text'])

    print(f"hand_written data loaded: {len(file_names)}")

    label_number = []
    for i in range(0, len(label_text)):
        label_number.append(cg.char_to_number(label_text[i]))

    return file_names, np.array(label_number, dtype=np.int)


