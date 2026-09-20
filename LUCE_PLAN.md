# Luce v0.2 — 확정 스펙 (2026-09-19)

Luce = Qwen 백본 위 1% 파라미터(LoRA + 결정 헤드)를 학습해 Choice/Score/Noul 확률 분포를 내는 결정 모델 **레시피**. 기본 백본은 `backbone: auto` = Qwen3-4B-Base (E15 사다리 실험에서 라벨 250~1,000 전 구간 4B > 1.7B > 0.6B; 작은 백본은 `model.backbone` 에 명시해 비용 선택). `trust_remote_code` / `backbone_overrides` / `label_overflow` 는 `model:` 에서 설정.
"과제를 문장으로 쓰면 → LLM이 데이터를 만들고 → 학습해서 → 서빙". 재현이 아니라 앞문.

## CLI
- `luce init "<과제 설명>" [--examples seeds.jsonl] [--teacher URL|MODEL] [--out luce.yaml]`
- `luce synth [--n N] [--teacher ...] [--writer ...] [--votes 3] [--dry-run] [--out data/synth]`
- `luce baseline --real real.jsonl [--n-perm 4]`  (v0 프롬프팅 하한)
- `luce train [--data ...] [--append review.jsonl] [--mode auto|bi|isolated|label] [--real real.jsonl] [--out ...]`
- `luce eval --checkpoint C --real real.jsonl [--synth ...] [--permute-seed N] [--dump ...]`
- `luce serve --checkpoint C [--port 8000] [--review review.jsonl --review-threshold 0.9]`
- 모든 명령은 `luce.yaml` 을 기본 설정으로 읽고 플래그가 덮어쓴다. 환경변수 `LUCE_*`.

## 결정 사항
1. 이름 `luce` (PyPI 비어 있음 확인). 모듈 `luce/`, `jevlocal` 은 한 버전 호환 shim.
2. 스승(teacher) 기본값 없음. `--teacher` 또는 `luce.yaml` 의 값이 없으면 에러 + Gateway / OpenAI / Ollama 세 예시를 그대로 복붙 가능하게 출력.
   형식: `URL|MODEL` (예: `https://ai-gateway.vercel.sh/v1|anthropic/claude-sonnet-5`, `https://api.openai.com/v1|gpt-5`, `http://localhost:11434/v1|qwen2.5:7b`).
   키는 환경변수(`LUCE_TEACHER_API_KEY`, 없으면 `OPENAI_API_KEY`/`AI_GATEWAY_API_KEY`).
3. `mode: auto`: 학습 데이터 0개 → 선택지 ≤26 이면 label, 초과면 isolated+lm-prior. 데이터 있음 → 모든 질문의 선택지가 닫힌 집합이면 bi, 아니면 isolated+lm-prior.
   auto 가 무엇을 왜 골랐는지(닫힌 집합 판정 결과 포함) 한 줄 출력.
4. review 임계값은 **보정 후 max-prob**, 기본 0.9. 엔트로피 confidence 는 부가 필드로 유지. `luce eval` 은 0.8/0.9/0.95 에서 coverage 와 그 구간 정확도(selective risk) 표 출력.
5. seeds 는 synth 스타일 참조 전용. `luce init --examples` 에 20개 이상 주면 절반 seeds / 절반 real 로 자동 분리.
6. `luce synth --dry-run`: 호출 수·예상 토큰·예상 비용 출력 후 중단.
7. 중복 제거 없음. 생성·라벨링·재라벨링·append 모두 동일하거나 유사한 내용이라는 이유로 관측을 삭제하지 않는다.
8. 실제 검증셋 강제: `--real` 없으면 경고 + 모든 수치 `[in-synth]` 표기. T 보정과 보고 숫자는 real 에서만.
9. 베스트 선택 = 보정 후 NLL, `last/` 저장, `--score-sigma 0` (이미 반영).
10. 보고: ECE + 노이즈 바닥 + 비율, 재조정 T, 타입별·source별 분해, reliability, selective risk.
11. 하지 말 것: 새 아키텍처 실험. 기존 코드를 앞문과 포장으로 묶는다.

## 완료 기준
새 환경에서 `pip install luce && luce init "..." && luce synth --n 500 && luce train && luce eval --real real.jsonl && luce serve` 가 돈다.
합성 티켓 과제에서 v0 대비 이기는 표가 README 에 있다. HF 에 체크포인트 5개가 `luce-examples/` 로 올라가 있다.

## 진행 상태 (2026-09-19)
- [x] 패키지 리네임 + shim, pyproject(`luce` 엔트리포인트), 새 venv 설치 검증
- [x] luce.yaml 스키마·로더, mode:auto + 근거 출력, teacher 필수 에러
- [x] init (teacher 초안 / 템플릿, seeds/real 티켓 단위 자동 분리, grid guard)
- [x] synth (페르소나·grid·선택적 정답 비율·near-miss·투표·dry-run·report)
- [x] train(--real 선택·보정, --append), eval([in-synth], 타입별 T, selective risk), serve(review 큐), baseline, convert
- [x] README(훅·퀵스타트·히어로 표·한계·약관·선행 작업), docs/ARCHITECTURE.md, 모델 카드 5개, 단위 테스트
- [ ] HF 업로드 `luce-examples/` (사용자 지시로 보류)
- [ ] openjev 열 측정, 시퀀스 예산 배처(기술 부채)

## 데모 (2026-09-20)
`luce serve` 가 `/demo` 를 같이 띄운다: 티켓 피드 → 질문 3개 확률 막대·지연·누적 비용, 0.9 미만은 검토 큐, 타이핑하면 실시간 재채점. `scripts/make_replay.py` 로 실제 서버 출력(답+지연)을 `luce/demo_replay.json` 에 저장하면 `/demo?replay=1`(또는 정적 서버의 `demo.html?replay=1`) 로 GPU 없이 같은 화면을 재생한다(화면에 replay 표시). `scripts/record_demo.py` 가 Playwright 로 녹화해 `media/live_triage.{mp4,gif}` 를 만든다.
