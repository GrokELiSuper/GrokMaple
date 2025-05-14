# 🛠️ 최종 수정본 (너가 요구한 구조 반영)
# 최신본 + 처음버전의 방식 통합 + 디버깅 추가

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

# ✅ 정답 문장 리스트
correct_sentences = [
    "바람이 산을 넘어간다", "꽃이 봄마다 핀다", "사과는 나무에서 자란다",
    "봄바람이 따듯하게 불어", "나무는 해를 받아자란다", "호수는 깊고 잔잔해",
    "해가 지면 어두워진다", "하늘은 넓고 푸르다", "달빛이 창가에 내린다",
    "물이 얼면 얼음이 된다", "나비가 꽃을 찾아 날아", "비가 내리고 땅이 젖는다",
    "사람은 밥을 먹고 산다", "강물이 흐른다", "별이 밤하늘에 빛나", "새가 하늘을 날아간다",
    "노을이 붉게 물들었어", "무궁화 꽃이 피었어!", "해가 뜨고 지는구나", "산에는 나무들이 많다",
    "이계던전 업무에 관해서", "엘나스 하급이계던전 입장 (900만 메소)", "이곳에서 나가고 싶습니다"
]

# ❌ 제외 키워드
ignore_phrases = ["중급", "누누코인"]

# 화면 캡처 영역 (QHD 왼쪽 위 1/4)
capture_region = {"top": 0, "left": 0, "width": 1280, "height": 720}

# 🧼 한글만 추출
def clean_korean(text):
    return re.sub(r'[^가-힣]', '', text)

# 📊 유사도 계산
def get_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# 📸 화면 캡처
def capture_screen():
    with mss.mss() as sct:
        shot = sct.grab(capture_region)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img.save("debug_capture.png")
        return np.array(img)

# 🎯 OCR 추출 + 필터링
def extract_easyocr_text(image_np):
    results = reader.readtext(image_np)
    lines = [res[1] for res in results if res[2] >= 0.6 and res[1].strip()]
    print("\n📝 OCR 추출 라인 결과:")
    for line in lines:
        print("-", line)
    return lines

# 🧹 OCR 줄 교정 적용 (선택사항)
def apply_correction(text_or_lines, is_line=False):
    correction_map = {
        "밥을": ["밤율", "밥율"],
        "빛나": ["빛다", "빚나"],
        "먹고": ["먹꼬", "먹구"],
        "얼음": ["얻음", "얼늠"],
        "하늘": ["하눌", "한늘"],
        "하급": ["하굽", "하겹"],
        "던전": ["더전", "터전"]
    }

    if is_line:
        line = text_or_lines
        for key, variants in correction_map.items():
            for v in variants:
                if v in line:
                    line = line.replace(v, key)
        return line  # 문자열 하나로 반환 (str)

    else:
        lines = text_or_lines
        corrected = []
        for line in lines:
            for key, variants in correction_map.items():
                for v in variants:
                    if v in line:
                        line = line.replace(v, key)
            corrected.append(line)
        return corrected  # 리스트로 반환 (list)



# 🔍 매칭 찾기 (줄별 + 전체 병합 둘 다)
def find_best_match(ocr_lines, threshold=0.5):
    best_match = None
    best_score = 0

    # ✨ 1차: 줄별로 직접 비교
    for line in ocr_lines:
        clean_line = clean_korean(line)
        for correct in correct_sentences:
            correct_clean = clean_korean(correct)
            score = get_similarity(clean_line, correct_clean)
            if correct_clean in clean_line:
                print(f"✅ [1차 줄 매칭] 완전 포함됨 → {correct}")
                return correct
            if score > best_score and score >= threshold:
                best_score = score
                best_match = correct

    # ✨ 2차: 전체 병합 후 비교
    full_text = "".join(ocr_lines)
    full_text_no_space = clean_korean(full_text)

    print(f"\n[📄 병합된 OCR 텍스트]:\n{full_text}")
    print(f"[🔍 병합 후 한글만 추출]:\n{full_text_no_space}\n")

    for correct in correct_sentences:
        correct_clean = clean_korean(correct)
        score = get_similarity(full_text_no_space, correct_clean)
        if correct_clean in full_text_no_space:
            print(f"✅ [2차 병합 매칭] 완전 포함됨 → {correct}")
            return correct
        if score > best_score and score >= threshold:
            best_score = score
            best_match = correct

    if best_match:
        print(f"🎯 최고 유사도 매칭 → {best_match} (점수: {best_score:.3f})")
    else:
        print("❌ 최종 매칭 실패")

    return best_match

# 🎯 문장 클릭
def click_sentence(image_np, target_sentence):
    results = reader.readtext(image_np)
    target = re.sub(r"[^가-힣]", "", target_sentence)

    found = False
    for bbox, text, _ in results:
        corrected_text = apply_correction(text, is_line=True)
        clean_text = re.sub(r"[^가-힣]", "", corrected_text)
        if target in clean_text:
            (tl, tr, br, bl) = bbox
            x = int((tl[0] + br[0]) / 2) + capture_region["left"]
            y = int((tl[1] + br[1]) / 2) + capture_region["top"]
            pyautogui.moveTo(x, y, duration=0.2)
            pyautogui.click()
            print(f"✅ 클릭 완료: {target_sentence} at ({x}, {y})")
            found = True
            break

    if not found:
        print(f"❌ [클릭 실패] '{target_sentence}'가 OCR 라인에 존재하지 않음")

# 🔁 전체 루프
def main_loop():
    print("[🎯] EasyOCR 거짓말 탐지기 루프 시작 (3초 간격)")
    try:
        while True:
            img_np = capture_screen()
            ocr_lines = extract_easyocr_text(img_np)
            ocr_lines = apply_correction(ocr_lines)

            matched = find_best_match(ocr_lines)
            if matched:
                print(f"🎯 정답 발견: {matched}")
                for _ in range(2):
                    click_sentence(img_np, matched)
                    time.sleep(0.5)
            else:
                print("❌ 정답 문장 없음")

            print("⏳ 3초 대기 후 다음 캡처...")
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n🛑 사용자 중지 요청 (Ctrl+C)")

if __name__ == "__main__":
    main_loop()
