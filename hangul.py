HANGUL_COUNT = 11172
ONSET_COUNT = 19  # 초성
NUCLEUS_COUNT = 21  # 중성
CODA_COUNT = 28  # 종성 (27자 + 공백)

ONSET_CORRECTOR = 588  # 중성 * 종성
NUCLEUS_CORRECTOR = 28  # 종성

# 수직 수평 모음 분리
NUCLEUS_V_TABLE = [1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 9, 0, 0, 5, 6, 9, 0, 0, 9, 9]
NUCLEUS_V_COUNT = 10

NUCLEUS_H_TABLE = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 5, 5, 0]
NUCLEUS_H_COUNT = 6

"""
    hangul number: 0 ~ 11171, UTF-16 기준 한글 완성형 순서에 따라서 지정된 번호
"""


def hangul_decode_by_number(number):
    onset_number = int(number / ONSET_CORRECTOR)
    number = number % ONSET_CORRECTOR

    nucleus_number = int(number / NUCLEUS_CORRECTOR)
    number = number % NUCLEUS_CORRECTOR

    coda_number = number

    return onset_number, nucleus_number, coda_number


def hangul_encode_to_number(onset_number, nucleus_number, coda_number):
    return onset_number * ONSET_CORRECTOR \
           + nucleus_number * NUCLEUS_CORRECTOR \
           + coda_number


def nucleus_separation(nucleus_number):
    return NUCLEUS_V_TABLE[nucleus_number], NUCLEUS_H_TABLE[nucleus_number]


def nucleus_integration(v_number, h_number):
    for i in range(0, NUCLEUS_COUNT):
        if NUCLEUS_V_TABLE[i] == v_number and \
           NUCLEUS_H_TABLE[i] == h_number:
            return i

    return -1
