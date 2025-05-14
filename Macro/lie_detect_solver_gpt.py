import mss
import easyocr
from PIL import Image
import numpy as np
import pyautogui
import time
import re
import difflib

# OCR 설정
reader = easyocr.Reader(['ko'], gpu=True)

# 정답 문장 리스트
correct_sentences = [
    "바람이 산을 넘어간다",
    "꽃이 봄마다 핀다",
    "사과는 나무에서 자란다",
    "봄바람이 따듯하게 불어",
    "나무는 해를 받아자란다",
    "호수는 깊고 잔잔해",
    "해가 지면 어두워진다",
    "하늘은 넓고 푸르다",
    "달빛이 창가에 내린다",
    "물이 얼면 얼음이 된다",
    "나비가 꽃을 찾아 날아",
    "비가 내리고 땅이 젖는다",
    "사람은 밥을 먹고 산다",
    "강물이 흐른다",
    "별이 밤하늘에 빛나",
    "새가 하늘을 날아간다"
]

correction_map = {
    "밥을": ["밤율", "밥율"],
    "빛나": ["빛다", "빚나"],
    "먹고": ["먹꼬", "먹구"],
    "얼음": ["얻음", "얼늠"],
    "하늘": ["하눌", "한늘"]
}

def generate_expanded_sentences(sentences, correction_map):
    expanded = set(sentences)
    for s in sentences:
        for key, variants in correction_map.items():
            if key in s:
                for v in variants:
                    expanded.add(s.replace(key, v))
    return list(expanded)

correct_sentences = generate_expanded_sentences(correct_sentences, correction_map)

# 화면 캡처 영역 (QHD 왼쪽 위 1/4)
capture_region = {"top": 0, "left": 0, "width": 1280, "height": 720}

def capture_screen():
    with mss.mss() as sct:
        shot = sct.grab(capture_region)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img.save("debug_capture.png")  # OCR 디버깅용 저장
        return np.array(img)

def extract_easyocr_text(image_np):
    results = reader.readtext(image_np)
    lines = [res[1] for res in results if res[1].strip()]
    return lines


def find_best_match(ocr_lines, threshold=0.5):
    full_text = "".join(ocr_lines)
    full_text_no_space = re.sub(r"[^가-힣]", "", full_text)

    print(f"\n[📄 병합된 OCR 텍스트]:\n{full_text}")
    print(f"[🔍 한글만 추출된 OCR 텍스트]:\n{full_text_no_space}\n")

    best_match = None
    best_score = 0

    for correct in correct_sentences:
        correct_clean = re.sub(r"[^가-힣]", "", correct)
        score = difflib.SequenceMatcher(None, full_text_no_space, correct_clean).ratio()

        print(f"[🔁 비교 대상] {correct} → {correct_clean}")
        print(f"📊 유사도 점수: {score:.3f}")

        if correct_clean in full_text_no_space:
            print(f"✅ [완전 포함됨] → 선택: {correct}\n")
            return correct

        if score > best_score and score >= threshold:
            best_score = score
            best_match = correct

    if best_match:
        print(f"\n🎯 유사도 최고 매칭 → {best_match} (점수: {best_score:.3f})")
    else:
        print("\n❌ 매칭된 문장이 없습니다.")

    return best_match


def click_sentence(image_np, target_sentence):
    results = reader.readtext(image_np)
    target = re.sub(r"[^가-힣]", "", target_sentence)

    for bbox, text, _ in results:
        clean_text = re.sub(r"[^가-힣]", "", text)
        if target in clean_text:
            (tl, tr, br, bl) = bbox
            x = int((tl[0] + br[0]) / 2) + capture_region["left"]
            y = int((tl[1] + br[1]) / 2) + capture_region["top"]
            pyautogui.moveTo(x, y, duration=0.2)
            pyautogui.click()
            print(f"✅ 클릭 완료: {target_sentence} at ({x}, {y})")
            return
    print("❌ 클릭 위치를 찾지 못했습니다.")

# ###
def save_ocr_log(ocr_lines):
    with open("ocr_log.txt", "a", encoding="utf-8") as f:
        f.write("[🧾 OCR 캡처 로그]\n")
        for line in ocr_lines:
            f.write(f"{line}\n")
        f.write("-" * 50 + "\n")


def apply_ocr_correction(text_lines, correction_map):
    corrected = []
    for line in text_lines:
        for key, variants in correction_map.items():
            for v in variants:
                if v in line:
                    line = line.replace(v, key)
        corrected.append(line)
    return corrected



def main():
    print("[🎯] EasyOCR 기반 거짓말 탐지기 자동 파훼 시작")
    time.sleep(2)

    img_np = capture_screen()
    ocr_lines = extract_easyocr_text(img_np)
    # 교정 필터 적용
    ocr_lines = apply_ocr_correction(ocr_lines, correction_map)
    # ocr 결과 로그 저장
    save_ocr_log(ocr_lines)

    print("📝 OCR 추출 결과:")
    for line in ocr_lines:
        print("-", line)

    matched = find_best_match(ocr_lines)
    if matched:
        print("🎯 정답 문장 발견:", matched)
        for i in range(2):
            click_sentence(img_np, matched)
            time.sleep(0.8)  # 클릭 간 약간의 딜레이 (필요에 따라 조절)
    else:
        print("❌ 정답 문장을 찾지 못했습니다.")

if __name__ == "__main__":
    main()
