# 레이블링 가이드라인 (MapleStory Worlds - Nunu World)

## ✅ 개정 사항 요약 (2025.05.14 통합)
- `shell_mid_boss` 클래스를 **두 개로 세분화**
  - `shell_mid_boss_closed`: 껍질이 남아있는 상태 → 초기 공격 패턴 (1 → r → 1 반복). 여기서 `1`은 무기 스왑 키이며, `r`이 실제 공격을 의미함. 첫 번째 `1`은 무기 스왑 후 공격을 위해, 마지막 `1`은 껍질이 까진 후 일반 공격을 위한 스왑.
  - `shell_mid_boss_opened`: 껍질이 제거된 상태 → shift 패턴 돌입 (4 + w + shift). mid_boss와 같은 공격 루틴.
- `skill_attack`을 **일반 공격 vs 빠른 연사 공격**으로 구분:
  - `skill_attack_normal`: 일반 스킬 (딜레이 존재)
  - `skill_attack_burst`: 빠른 연사 스킬 (연속 발사형)
- `jump_action` 클래스 **삭제됨** (더 이상 사용하지 않음)
- `ladder_rope`와 `portal` 클래스는 **겹침 시에도 각각 개별 box로 명확히 분리**해서 라벨링
- `ladder_rope`는 마지막 보상맵에서 label하지 않는다.
- `player_dot`은 YOLO fine-tuning으로 분리 관리되므로 레이블링은 유지하되 **crop 이미지 기반 학습**을 병행함

## ✅ 일반 레이블링 원칙
- 가장 가까운 시간에 보이는 내용을 기준으로 라벨링
- 가능한 한 Tight하게 bbox 구성
- 객체가 일부만 보여도 레이블링 대상
- 설치형 스킬과 보스의 겹침은 허용, 개별 라벨링
- 미니맵이 꺼진 경우 player_dot 레이블링 중단
- 쓰러진 몬스터는 레이블링 제외
- 텔레포트는 skill_attack으로 간주 (이펙트 명확할 경우)

## 🎯 현재 클래스 목록 (총 15개)
| class name              | 설명 |
|-------------------------|------|
| npc                    | NPC 캐릭터. 주요 상호작용 대상. |
| player_dot             | 플레이어 위치 표시. 미니맵 기준. crop 및 fine-tune 대상 |
| skill_attack_normal    | 일반 스킬 이펙트. 빛, 폭발 등 |
| skill_attack_burst     | 빠른 연사형 스킬 이펙트 |
| skill_install          | 설치형 스킬. 고정형 오브젝트 |
| projectile_attack      | 발사형 투사체 스킬. reward_box, hit_object 공격에 사용 |
| monster                | 일반 몬스터 (기본 공격) |
| mid_boss               | 껍질 제거된 중간보스 (4 + w + shift) |
| shell_mid_boss_closed  | 껍질이 남아있는 중간보스 (1 → r → 1 패턴. 무기 스왑과 공격 포함) |
| shell_mid_boss_opened  | 껍질 제거 후 shift 패턴 돌입 상태 (mid_boss와 동일 루틴) |
| portal                 | 포탈. 맵 이동 가능 지점 |
| hit_object             | 공격 가능한 오브젝트 (보상맵 마지막 등장) |
| reward_box             | 클리어 보상 상자 |
| item_drop              | 아이템 드랍. 펫/Z키로 획득 가능 |
| ladder_rope            | 사다리 또는 로프 구조물 |

## 🧭 참고 사항
- 겹쳐 있는 객체(예: 몬스터)는 개별 라벨링
- 설치형 스킬과 보스 겹침 허용
- rope/ladder는 일부 가려져도 전체 box로 라벨링
- YOLO 라벨 포맷 기준: `class_id cx cy w h` (정규화 값)

> 레이블링 정확도는 모델 성능에 직결됩니다. 최신 기준을 항상 반영해주세요.