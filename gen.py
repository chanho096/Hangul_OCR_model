import logging
import glob
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def generate_input_data(cg, tg, shape, size):
    label = np.random.randint(low=0, high=cg.get_character_count(), size=size)
    font = np.random.randint(low=0, high=tg.get_font_count(), size=size)
    font_size = np.random.randint(low=80, high=150, size=size)

    font_color_R = np.random.randint(low=0, high=256, size=size)
    font_color_G = np.random.randint(low=0, high=256, size=size)
    font_color_B = np.random.randint(low=0, high=256, size=size)

    input_data = np.zeros(shape=(size, shape[0], shape[1], shape[2]), dtype=np.uint8)

    for i in range(0, size):
        char = cg.number_to_char(label[i])
        image = tg.char_to_image(char, font_size[i], font[i],
                                 (font_color_R[i], font_color_G[i], font_color_B[i]))
        image = image.resize((shape[0], shape[1]))
        input_data[i] = np.array(image, dtype=np.uint8).reshape(shape)

    return input_data, label


class CharacterGenerator:
    """
        학습에 사용될 문자들을 정수로 mapping 하는 기능을 수행한다.

        문자들은 utf-16 형식이다.
        block 단위로 사용되는 문자 범위를 지정할 수 있다.
    """

    _ASCII_NUMBER_FIRST = 0x0030
    _ASCII_NUMBER_LAST = 0x0039
    _ASCII_UPPER_ALPHABET_FIRST = 0x0041
    _ASCII_UPPER_ALPHABET_LAST = 0x005A
    _ASCII_LOWER_ALPHABET_FIRST = 0x0061
    _ASCII_LOWER_ALPHABET_LAST = 0x007A
    _HANGUL_SYLLABLES_FIRST = 0xAC00
    _HANGUL_SYLLABLES_LAST = 0xD7A3

    def __init__(self):
        self._blocks = [self._HANGUL_SYLLABLES_FIRST,
                        self._ASCII_NUMBER_FIRST,
                        self._ASCII_UPPER_ALPHABET_FIRST,
                        self._ASCII_LOWER_ALPHABET_FIRST]
        self._blocks_size = [int(self._HANGUL_SYLLABLES_LAST - self._HANGUL_SYLLABLES_FIRST + 1),
                             int(self._ASCII_NUMBER_LAST - self._ASCII_NUMBER_FIRST + 1),
                             int(self._ASCII_UPPER_ALPHABET_LAST - self._ASCII_UPPER_ALPHABET_FIRST),
                             int(self._ASCII_LOWER_ALPHABET_LAST - self._ASCII_LOWER_ALPHABET_FIRST)]
        self._blocks_count = len(self._blocks_size)

    def get_character_count(self):
        """
            생성되는 모든 Character 개수를 반환한다.
        """
        number = 0
        for i in range(0, self._blocks_count):
            number += self._blocks_size[i]
        return number

    def number_to_char(self, number):
        """
            지정된 번호의 Character 를 반환한다.
            
            지정된 번호를 초과하는 경우, 오류 출력 및 null 문자 반환
        """
        for i in range(0, self._blocks_count):
            if number < self._blocks_size[i]:
                # 정상적으로 문자 변환하여 출력 (utf-16)
                utf_code = self._blocks[i] + number
                character = chr(utf_code).encode('utf-16', 'surrogatepass').decode('utf-16')
                return character
            else:
                # 다음 블록에서 문자를 찾는다.
                number -= self._blocks_size[i]

        # 번호 범위 초과, 탐색 실패
        logging.error("Number range exceeded in character generator")
        return chr(0)

    def char_to_number(self, char):
        """
            지정된 Character 의 번호를 반환한다.
        """
        utf_code = char.encode('utf-16', 'surrogatepass')
        utf_number = ord(utf_code.decode('utf-16'))

        number = 0
        for i in range(0, self._blocks_count):
            if self._blocks[i] <= utf_number < self._blocks[i] + self._blocks_size[i]:
                # 정상적으로 번호 변환하여 출력 (utf-16)
                number = number + utf_number - self._blocks[i]
                return number
            else:
                # 다음 블록에서 문자를 찾는다.
                number += self._blocks_size[i]

        # Unicode 범위 초과, 탐색 실패
        logging.error("Unicode range exceeded in character generator")
        return -1


class TextImageGenerator:
    """
        Character 를 입력으로 받아서, Text Image 를 반환한다.

        Text Image 는 임의 지정된 true type font 로 생성할 수 있다.
        Image 를 생성하기 위하여 Pillow 라이브러리를 사용한다.
        Pillow Image 객체를 반환한다.
    """
    def __init__(self, font_dir):
        self.font_dir = None
        self.font_list = None
        self.font_count = None

        if not os.path.exists(font_dir):
            logging.error("Invalid font path")
        else:
            font_file_list = glob.glob(os.path.join(font_dir, '*.ttf'))
            font_file_list.extend(glob.glob(os.path.join(font_dir, '*.ttc')))

            font_list = []
            for font_file in font_file_list:
                font_list.append(ImageFont.truetype(font_file))
            font_count = len(font_list)

            if font_count < 1:
                logging.error("Failed to load font files")
            else:
                # font 로드 성공
                self.font_dir = font_dir

        if self.font_dir is None:
            # font 로드 실패
            logging.error("Failed to initialize text image generator")
        else:
            # font 로드 성공
            logging.info("Font file loaded successfully")
            self.font_list = font_list
            self.font_count = font_count

    def get_font_count(self):
        if self.font_dir is None:
            return 0
        else:
            return self.font_count

    def char_to_image(self, char, size, font_number, color):
        if self.font_dir is None:
            logging.error("Text Image generator is not initialized")
            return None
        elif not font_number < self.font_count:
            logging.error("Invalid font number")
            return None

        # Pillow 라이브러리를 이용하여 문자를 그린다.
        image = Image.new('RGB', (size+50, size+50), color=(0, 0, 0))
        d = ImageDraw.Draw(image, mode="RGB")
        font = self.font_list[font_number].font_variant(size=size)
        d.text((0, 0), char, font=font, fill=color)

        # 이미지를 글자 크기에 알맞게 자른다.
        image_box = image.getbbox()
        cropped_image = image.crop(image_box)

        return cropped_image



