# Last-Pass End Location Prediction (preprocess + AutoGluon)

이 저장소는 **축구 이벤트 로그**에서 각 `game_episode`의 **마지막 Pass**를 예측 대상으로 삼아, 해당 패스의 **도착 좌표 `(end_x, end_y)`**를 회귀로 예측하는 파이프라인입니다.

- `preprocess.ipynb`  
  → **train/test 이벤트 로드 → 마지막 Pass 샘플 생성(에피소드당 1행) → 피처 생성 → k_prev 버전별 저장**
- `train_autogluon.ipynb`  
  → **GroupKFold(game_id) OOF 평가(유클리드) 기반으로 k_prev/피처/메트릭/분기/후처리를 선택하고 최종 제출 파일 생성**

> ⚠️ 주의: 본 문서는 제공된 노트북 코드(preprocess/train_autogluon)의 동작을 “그대로 설명”합니다.  
> 데이터 스키마(컬럼 의미, episode/period 정의 등)가 대회/데이터 문서와 다를 경우 결과가 달라질 수 있습니다(확실하지 않음).

---

## 1. 문제 정의(이 파이프라인 관점)

- **샘플 단위**: `game_episode` 1개당 1샘플  
- **예측 대상**: 해당 episode의 **마지막 Pass 이벤트** 1개  
- **타깃 라벨**: 마지막 Pass의 `(end_x, end_y)`
- **평가(모델 선택 기준)**: 평균 유클리드 거리

\[
\text{score} = \frac{1}{N}\sum_{i=1}^N \sqrt{(x_i-\hat x_i)^2 + (y_i-\hat y_i)^2}
\]

AutoGluon 내부의 `eval_metric(rmse/mae)`는 **학습 과정에서의 내부 기준**이고, 최종 후보 비교는 항상 **유클리드 OOF score**로 합니다.

---

## 2. 디렉토리 구조
├── data/  
│ ├── train.csv  
│ ├── test.csv # index: game_id, game_episode, path  
│ └── test/.../{game_episode}.csv  
├── artifacts/ # preprocess outputs  
│ ├── features_train_k3.parquet  
│ ├── labels_train_k3.parquet  
│ ├── features_test_k3.parquet  
│ ├── ...  
│ └── test_index.csv  
├── models_ag/ # trained predictors + meta  
│ ├── predictor_endx...  
│ ├── predictor_endy...  
│ └── model_meta.json  
├── ag_tmp/ # fold-level temporary predictors (auto cleaned)  
├── oof_cache/ # cached OOF 결과 (pkl)  
└── submission.csv

---

## 3. preprocess.ipynb 설명

### 3.1 입력

- `data/train.csv` : train 이벤트 로그(마지막 Pass의 `end_x,end_y` 라벨 포함)
- `data/test.csv` : test 인덱스 파일
  - 필수 컬럼: `game_id`, `game_episode`, `path`
- `data/test/.../{game_episode}.csv` : episode별 이벤트 로그

### 3.2 출력

k_prev = 3,5,7,10 등에 대해 아래가 생성됩니다.

- `artifacts/features_train_k{K}.parquet`
- `artifacts/labels_train_k{K}.parquet`
- `artifacts/features_test_k{K}.parquet`
- `artifacts/test_index.csv`

### 3.3 핵심 아이디어: “마지막 Pass”만 학습 샘플로 만들기

1) 전체 이벤트를 episode 내부 시간 순으로 정렬합니다.

- 정렬 키:
  - `game_episode`, `period_id`, `time_seconds`, `action_id`

2) `type_name == "Pass"`인 이벤트들 중, episode별 **마지막 1개**만 `is_target_pass=True`로 표시합니다.

3) 최종적으로 `is_target_pass=True`인 행만 남겨서  
   **episode 당 1행짜리 학습 데이터(X)와 라벨(y)** 를 만듭니다.

### 3.4 공간(Spatial) 피처: start 좌표 기준 파생

`start_x, start_y`를 기준으로 아래를 추가합니다.

- 골까지 거리/각도:
  - `start_dist_to_goal`, `start_angle_to_goal`
- 터치라인/엔드라인까지 거리:
  - `start_dist_to_sideline`, `start_dist_to_endline`
- 스케일/중앙 기준 피처:
  - `start_x_ratio`, `start_y_centered_abs`

> 경기장 상수:
> - PITCH_X=105, PITCH_Y=68  
> - GOAL_X=105, GOAL_Y=34

### 3.5 k_prev(히스토리 윈도우) 피처: 직전 k개 이벤트를 shift로 붙이기

`k_prev`는 **마지막 Pass 직전 이벤트를 몇 개까지 피처로 볼지**를 의미합니다.

- `prev1_*` : 직전 1개 이벤트의 정보  
- `prev2_*` : 직전 2개 이벤트의 정보  
- ...
- `prevK_*` : 직전 K개 이벤트의 정보

`base_cols`에 대해 `groupby(game_episode).shift(i)`로 생성합니다.

- base_cols:
  - `type_name`, `result_name`, `start_x`, `start_y`, `end_x`, `end_y`, `time_seconds`

추가로 이동량 파생 피처도 생성합니다.

- `prev{i}_dx = prev{i}_end_x - prev{i}_start_x`
- `prev{i}_dy = prev{i}_end_y - prev{i}_start_y`
- `prev{i}_dist_move = sqrt(dx^2 + dy^2)`

그리고 k개를 요약한 집계 피처를 만듭니다.

- `prevk_sum_dx`
- `prevk_sum_abs_dy`
- `prevk_mean_dx`
- `prevk_lateral_ratio = sum_abs_dy / (abs(sum_dx)+eps)`

### 3.6 episode 요약 피처(타깃 이전 구간)

episode 내에서 **타깃(마지막 Pass) 이전** 이벤트들의 누적 움직임을 요약합니다.

- `ep_len_before` : 타깃 이전 이벤트 수
- `ep_time_span` : `max(time_seconds) - min(time_seconds)`
- `ep_sum_dx_before` : 타깃 이전 이벤트들의 dx 합
- `ep_sum_abs_dy_before` : 타깃 이전 이벤트들의 |dy| 합

### 3.7 최종 반환

- `X`: `[game_episode] + feature_cols`  
- `y`: train에 한해 `[game_episode, end_x, end_y]`

---

## 4. train_autogluon.ipynb 설명

이 노트북은 preprocess 산출물(`features_train_k*.parquet`)을 사용하여,  
**OOF(Out-of-Fold) 유클리드 거리 기준으로** 설정을 선택하고 최종 제출을 생성합니다.

### 4.1 왜 GroupKFold(game_id)인가?

같은 경기(`game_id`)의 episode가 train/valid로 섞이면,  
경기 고유 패턴이 누출되어 성능이 과대평가될 수 있습니다.

따라서 `groups = game_id`로 GroupKFold를 적용해  
**같은 게임은 항상 같은 fold에만 존재**하도록 만듭니다.

### 4.2 end_x/end_y 분리 학습

AutoGluon Tabular은 기본적으로 “단일 타깃 회귀”에 최적화되어 있으므로,
- `end_x` predictor 1개
- `end_y` predictor 1개
를 각각 학습하고 마지막에 `(x,y)`로 합쳐 평가합니다.

### 4.3 전처리 일관화: object 결측치 처리

object 타입 컬럼의 결측치는 `fill_object_missing()`으로 `"MISSING"`으로 통일합니다.

이 처리는 아래 모든 구간에 동일하게 적용됩니다.

- OOF 생성(train/valid)
- feature importance 계산(pruning)
- 최종 전체 학습
- 테스트 예측

> 주의: 이 함수는 `dtype == object`만 대상으로 합니다.  
> categorical dtype이 섞여 있다면 동일하게 처리되지 않을 수 있습니다(확실하지 않음).

### 4.4 임시 predictor 폴더 정리(try/finally)

fold별 AutoGluon 학습 결과는 임시 폴더에 저장되며,
에러가 나더라도 `finally`에서 폴더를 지워 디스크 오염을 방지합니다.

- `ag_tmp/end_x_fold{fold}`
- `ag_tmp/end_y_fold{fold}`
- importance/branching도 유사한 임시 경로 사용

### 4.5 OOF 캐시(재실행 비용 절감)

OOF score 계산은 매우 비싸기 때문에,
`oof_cache/`에 `(score, oof_pred)`를 pkl로 저장해 재사용합니다.

캐시 키는 다음으로 구성됩니다.

- `k_prev`
- `feature columns`의 md5 hash
- x/y metric
- cv split 수
- tag (ksearch/cand 등)

이 덕분에 같은 설정을 반복 실행할 때 학습을 다시 하지 않습니다.

---

## 5. 파이프라인 단계별 흐름(STEP 1~6)

### STEP 1) k_prev 탐색 (cheap preset + OOF 캐시)

**목적**: k_prev 후보(3,5,7,10) 중 최적의 히스토리 길이를 선택

- 빠른 탐색용 preset: `SEARCH_PRESETS_K = "medium_quality"`
- 각 k_prev에 대해:
  - `ag_cv_score_xy()`로 end_x/end_y OOF 예측 생성
  - 클리핑 적용 후 유클리드 OOF score 계산
  - 가장 낮은 score의 k_prev를 채택

> 왜 cheap preset인가?  
> k_prev 후보를 빠르게 걸러내고, 이후 단계에서 더 좋은 preset으로 정밀 비교하기 위함입니다.

---

### STEP 2) pruning (CV-train importance top-N)

**목적**: 피처 수를 줄여 과적합/학습시간을 완화할 가능성을 탐색

- 각 fold에서 **train subset으로만** predictor를 학습
- train subset에서 feature importance를 계산
- fold별 importance 평균으로 top-N 피처 선택

> 누수 완화 포인트  
> valid fold 정보로 importance를 계산하지 않도록 “CV-train에서만” importance를 누적합니다.

---

### STEP 3) 후보 비교 (baseline vs pruned, metric 조합)

**목적**: 피처셋/내부 metric 조합을 바꿨을 때 유클리드 OOF가 개선되는지 확인

후보 예시:
- baseline + (x=rmse, y=rmse)
- baseline + (x=rmse, y=mae)
- pruned + (x=rmse, y=rmse)
- pruned + (x=rmse, y=mae)

각 후보는 OOF 캐시를 사용해 중복 학습을 방지합니다.  
최종 선택 기준은 **유클리드 OOF 최소**입니다.

---

### STEP 4) Branching OOF (result_name 기반 분기)

**목적**: `result_name`이 “Successful/Unsuccessful”에 따라 도착점 분포가 달라질 수 있으므로,
분기 학습이 OOF를 개선하는지 확인합니다.

- train fold에서:
  - 성공 subset으로 end_x/end_y 모델 학습(2개)
  - 실패 subset으로 end_x/end_y 모델 학습(2개)
- valid fold에서:
  - 각 샘플의 `result_name`으로 라우팅하여 해당 모델로만 예측(효율)

안전장치:
- 한쪽 표본이 너무 적으면(`min_side=50`) 분기 대신 전체 모델로 fallback

> ⚠️ 실전 제약  
> 테스트 데이터에 `result_name`이 없다면 branching은 사용할 수 없으며, 코드에서 자동으로 비활성화합니다.

---

### STEP 5) 후처리(postprocess) 튜닝 (OOF grid search)

**목적**: 예측 좌표를 간단히 물리적으로 보정해 유클리드 score를 낮출 가능성을 탐색

후처리 규칙:
- x: 시작점 대비 전진/후진 변위를 `forward_scale`로 스케일
- y: 중앙선(GOAL_Y=34) 기준으로 편차를 `lateral_shrink`로 축소/확대

grid 예시:
- `forward_scale`: [0.90, 0.95, 1.00, 1.05]
- `lateral_shrink`: [0.80, 0.90, 1.00]

OOF에 후처리를 적용했을 때 score가 실제로 내려가면 `use_postprocess=True`.

---

### STEP 6) 최종 학습 + submission 생성

최종 구성(`final config`)을 확정한 뒤 전체 데이터를 사용해 학습합니다.

- branching 미사용:
  - end_x predictor 1개 + end_y predictor 1개 학습
- branching 사용(가능할 때):
  - 성공/실패 각각 end_x/end_y predictor 학습(총 4개)
  - 테스트를 result_name으로 라우팅

마지막에:
- 예측값을 경기장 범위로 클리핑
- (선택) 후처리 적용
- `submission.csv` 저장:
  - `game_episode`, `end_x`, `end_y`

또한 `models_ag/model_meta.json`에 아래를 저장합니다.
- 선택된 k_prev
- feature set 이름 및 feature columns
- metric 조합
- branching/postprocess 여부 및 파라미터
- OOF score(후처리 전/후)

---

## 6. 실행 순서

### 6.1 preprocess 실행
1) `data/`에 입력 파일 배치
2) `preprocess.ipynb` 실행
3) `artifacts/`에 k_prev별 parquet 생성 확인

### 6.2 학습/제출 생성
1) `train_autogluon.ipynb` 실행
2) 결과:
   - `submission.csv`
   - `models_ag/model_meta.json`
   - (선택) `oof_cache/*.pkl` 누적

---

## 7. 설계 의도 요약(왜 이렇게 했는가)

- **마지막 Pass만 예측**: 목표가 명확(episode당 1개), 데이터/학습 단순화, 제출 포맷 정렬이 쉬움
- **k_prev 다중 버전**: “얼마나 과거를 볼지”가 성능에 큰 영향을 주므로 탐색
- **GroupKFold(game_id)**: 경기 단위 누수 방지 및 더 현실적인 일반화 평가
- **end_x/end_y 분리 학습**: AutoGluon Tabular의 단일 타깃 학습 구조에 맞춤
- **OOF 캐시**: 후보 비교 시 반복 학습 비용을 크게 절감
- **CV-train importance pruning**: 피처 수를 줄이는 전략을 (누수 완화 형태로) 실험
- **branching(result_name)**: 성공/실패 분포 차이를 반영해 모델을 분리 학습
- **후처리 grid tuning**: 예측 좌표의 물리적 보정을 간단한 파라미터로 탐색

---

## 8. 알려진 제약/체크 포인트

- 테스트에 `result_name`이 없으면 branching은 자동으로 꺼집니다(코드상 확정).
- 정렬 키에 `period_id`가 포함되어 있어, episode가 period를 넘나드는 데이터에서
  순서 정의가 의도와 다를 수 있습니다(확실하지 않음: 데이터 정의 필요).
- `fill_object_missing()`은 object dtype만 처리합니다(categorical dtype은 별도 확인 필요, 확실하지 않음).

---

## 9. 참고: 주요 하이퍼/프리셋

- k_prev 탐색: `SEARCH_PRESETS_K = "medium_quality"`
- 후보 비교/importance: `SEARCH_PRESETS = "good_quality"`
- 최종 학습: `FINAL_PRESETS = "best_quality"`
- CV: `N_SPLITS = 5`
- branching 최소 표본: `min_side = 50`
- 후처리 grid:
  - forward_scale: 0.90~1.05
  - lateral_shrink: 0.80~1.00

---

## 10. 재현성 팁

- `oof_cache/`와 `models_ag/`는 실험 재실행 시 성능/설정 추적에 유용합니다.
- 동일한 후보를 반복 실행할 경우 OOF 캐시가 켜져 있으면 학습이 생략됩니다.
- GPU 환경에서 AutoGluon 동작은 버전/환경에 따라 달라질 수 있습니다(확실하지 않음).

---