HANGUL_COUNT = 11172
ONSET_COUNT = 19  # 초성
NUCLEUS_COUNT = 21  # 중성
CODA_COUNT = 28  # 종성 (27자 + 공백)

ONSET_CORRECTOR = 588  # 중성 * 종성
NUCLEUS_CORRECTOR = 28  # 종성

"""
    hangul number: 0 ~ 11171, UTF-16 기준 한글 완성형 순서에 따라서 지정된 번호
"""


def hangul_decode_by_number(number):
    onset_number = int(number / ONSET_CORRECTOR)
    number = number % ONSET_CORRECTOR

    nucleus_number = int(number / NUCLEUS_CORRECTOR)
    number = number % NUCLEUS_CORRECTOR

    coda_number = number
    if onset_number >= ONSET_COUNT or nucleus_number >= NUCLEUS_COUNT or coda_number >= CODA_COUNT:
        print("???")

    return onset_number, nucleus_number, coda_number


def hangul_incode_to_number(onset_number, nucleus_number, coda_number):
    return onset_number * ONSET_CORRECTOR \
           + nucleus_number * NUCLEUS_CORRECTOR \
           + coda_number
