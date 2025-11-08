# Vertex AI 크레딧 차감 확인 방법

## 1. Google Cloud Console에서 확인

### 방법 1: Vertex AI 사용량 대시보드

1. **Google Cloud Console 접속**
   ```
   https://console.cloud.google.com/
   ```

2. **프로젝트 선택**
   - 상단에서 프로젝트 선택: `astute-coda-477522-r1`

3. **Vertex AI 사용량 확인**
   ```
   Navigation Menu → Vertex AI → Dashboard
   또는
   https://console.cloud.google.com/vertex-ai/dashboard?project=astute-coda-477522-r1
   ```

4. **사용량 보기**
   - "Usage" 또는 "Metrics" 탭 클릭
   - Gemini API 호출 횟수 확인
   - 토큰 사용량 확인

### 방법 2: Billing 페이지에서 확인 (가장 정확)

1. **Billing 페이지 접속**
   ```
   Navigation Menu → Billing → Reports
   또는
   https://console.cloud.google.com/billing/
   ```

2. **프로젝트 필터링**
   - Project: `astute-coda-477522-r1` 선택
   - Service: "Vertex AI API" 또는 "Generative AI" 선택

3. **사용량 확인**
   - 시간대별 사용량 그래프 확인
   - 비용 항목 확인
   - SKU (Stock Keeping Unit) 확인:
     - `Gemini 2.5 Flash - Input Characters`
     - `Gemini 2.5 Flash - Output Characters`

4. **필터 설정**
   - Time range: "Last 24 hours" 또는 "Today"
   - Group by: "SKU" 또는 "Service"

### 방법 3: Cloud Monitoring

1. **Monitoring 페이지 접속**
   ```
   Navigation Menu → Monitoring → Metrics Explorer
   또는
   https://console.cloud.google.com/monitoring?project=astute-coda-477522-r1
   ```

2. **메트릭 선택**
   - Resource type: "Vertex AI API"
   - Metric: "Request count" 또는 "Token count"

3. **시간대 확인**
   - 최근 1시간 또는 6시간 선택
   - 호출 시간대와 일치하는지 확인

## 2. gcloud CLI로 확인

### Billing Account 확인

```bash
# Billing account 목록 조회
gcloud billing accounts list

# 특정 프로젝트의 billing 정보 확인
gcloud billing projects describe astute-coda-477522-r1
```

### Vertex AI API 사용량 확인

```bash
# API 사용량 확인 (최근 24시간)
gcloud logging read "resource.type=aiplatform.googleapis.com" \
  --project=astute-coda-477522-r1 \
  --limit=100 \
  --format=json

# Gemini API 호출 로그 확인
gcloud logging read "resource.type=aiplatform.googleapis.com AND protoPayload.methodName=~'generateContent'" \
  --project=astute-coda-477522-r1 \
  --limit=50 \
  --format="table(timestamp, protoPayload.methodName, protoPayload.request.model)"
```

### 비용 추정 확인

```bash
# 현재 월 비용 추정치 확인 (BigQuery 필요)
gcloud alpha billing accounts get-iam-policy YOUR_BILLING_ACCOUNT_ID
```

## 3. 프로그래밍 방식 확인 (Python)

### Cloud Billing API 사용

```python
from google.cloud import billing_v1
from datetime import datetime, timedelta

def get_billing_usage(project_id: str):
    """프로젝트의 billing 사용량 조회"""

    # Billing 클라이언트 생성
    client = billing_v1.CloudCatalogClient()

    # 서비스 목록 조회
    services = client.list_services()

    for service in services:
        if 'vertex' in service.display_name.lower():
            print(f"Service: {service.display_name}")
            print(f"Service ID: {service.name}")

            # SKU 목록 조회
            parent = service.name
            skus = client.list_skus(parent=parent)

            for sku in skus:
                if 'gemini' in sku.description.lower():
                    print(f"  SKU: {sku.description}")
                    print(f"  Category: {sku.category.resource_family}")

# 사용
get_billing_usage("astute-coda-477522-r1")
```

### Cloud Monitoring API 사용

```python
from google.cloud import monitoring_v3
from datetime import datetime, timedelta

def get_vertex_ai_metrics(project_id: str):
    """Vertex AI 메트릭 조회"""

    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    # 시간 범위 설정 (최근 1시간)
    now = datetime.utcnow()
    interval = monitoring_v3.TimeInterval({
        "end_time": {"seconds": int(now.timestamp())},
        "start_time": {"seconds": int((now - timedelta(hours=1)).timestamp())},
    })

    # 메트릭 조회
    results = client.list_time_series(
        request={
            "name": project_name,
            "filter": 'resource.type = "aiplatform.googleapis.com/Endpoint"',
            "interval": interval,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        }
    )

    for result in results:
        print(f"Metric: {result.metric.type}")
        print(f"Resource: {result.resource.labels}")
        for point in result.points:
            print(f"  Time: {point.interval.end_time}")
            print(f"  Value: {point.value.int64_value or point.value.double_value}")

# 사용
get_vertex_ai_metrics("astute-coda-477522-r1")
```

## 4. 우리 테스트 결과 확인 방법

### 예상 사용량 (30개 동시 호출 테스트 기준)

```
입력 토큰:  1,078개
출력 토큰:  2,875개
총 토큰:    3,953개

예상 비용 (Gemini 2.5 Flash 기준):
- Input:  1,078 tokens × $0.000015/1K tokens = $0.00002
- Output: 2,875 tokens × $0.00006/1K tokens  = $0.00017
- 총:     약 $0.00019 (호출당)
```

### 확인 순서

1. **즉시 확인 (5분 이내)**
   - Cloud Console → Vertex AI → Dashboard
   - Recent requests 확인
   - API call count 확인

2. **상세 확인 (1시간 이내)**
   - Cloud Console → Billing → Reports
   - Time range: "Last hour"
   - Service: "Vertex AI API"
   - 토큰 사용량 및 비용 확인

3. **로그 확인**
   - Cloud Console → Logging → Logs Explorer
   - Query:
     ```
     resource.type="aiplatform.googleapis.com"
     protoPayload.methodName="generateContent"
     timestamp>="2025-11-08T08:00:00Z"
     ```

## 5. 실시간 모니터링 스크립트

```python
#!/usr/bin/env python3
"""
Vertex AI 실시간 사용량 모니터링 스크립트
"""

import os
from google.cloud import logging
from datetime import datetime, timedelta

def monitor_vertex_ai_usage(project_id: str, hours: int = 1):
    """Vertex AI API 호출 로그 모니터링"""

    client = logging.Client(project=project_id)

    # 로그 필터 설정
    filter_str = f'''
    resource.type="aiplatform.googleapis.com"
    protoPayload.methodName="google.cloud.aiplatform.v1.PredictionService.GenerateContent"
    timestamp>="{(datetime.utcnow() - timedelta(hours=hours)).isoformat()}Z"
    '''

    print(f"📊 Vertex AI 사용량 모니터링 (최근 {hours}시간)")
    print("=" * 80)

    total_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # 로그 조회
    for entry in client.list_entries(filter_=filter_str, page_size=100):
        total_calls += 1

        # 토큰 정보 추출 (로그 구조에 따라 다를 수 있음)
        if hasattr(entry, 'payload') and entry.payload:
            print(f"\n[{entry.timestamp}]")
            print(f"  Method: {entry.payload.get('methodName', 'N/A')}")

            # Usage metadata 확인
            usage = entry.payload.get('response', {}).get('usageMetadata', {})
            if usage:
                input_tokens = usage.get('promptTokenCount', 0)
                output_tokens = usage.get('candidatesTokenCount', 0)

                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                print(f"  Input tokens: {input_tokens}")
                print(f"  Output tokens: {output_tokens}")

    # 요약 출력
    print("\n" + "=" * 80)
    print(f"📈 요약:")
    print(f"  총 호출 횟수: {total_calls}회")
    print(f"  총 입력 토큰: {total_input_tokens:,}개")
    print(f"  총 출력 토큰: {total_output_tokens:,}개")
    print(f"  총 토큰: {total_input_tokens + total_output_tokens:,}개")

    # 예상 비용 (Gemini 2.5 Flash 기준)
    input_cost = (total_input_tokens / 1000) * 0.000015
    output_cost = (total_output_tokens / 1000) * 0.00006
    total_cost = input_cost + output_cost

    print(f"\n💰 예상 비용:")
    print(f"  Input: ${input_cost:.6f}")
    print(f"  Output: ${output_cost:.6f}")
    print(f"  Total: ${total_cost:.6f}")

if __name__ == "__main__":
    monitor_vertex_ai_usage("astute-coda-477522-r1", hours=1)
```

## 6. 빠른 확인 방법 (권장)

가장 빠르고 정확한 확인 방법:

```bash
# 1. 브라우저에서 바로 접속
open "https://console.cloud.google.com/billing/reports?project=astute-coda-477522-r1"

# 2. 또는 Vertex AI Dashboard
open "https://console.cloud.google.com/vertex-ai/dashboard?project=astute-coda-477522-r1"
```

그리고 페이지에서:
1. Time range: "Last hour" 선택
2. Service: "Vertex AI" 필터
3. SKU에서 "Gemini 2.5 Flash" 찾기
4. 토큰 수와 비용 확인

---

**참고**: Billing 데이터는 실시간이 아니라 몇 분~1시간 정도 지연될 수 있습니다.
가장 빠른 확인은 Cloud Console의 Vertex AI Dashboard입니다.
