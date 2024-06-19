# electra-pretrain-rewrite
 사내에서 electra 모델을 사전학습했던 코드를 재작성합니다. 데이터는 명세만 공개합니다.

## data structure
```
{
    "applicate_number": int, 특허 출원번호
    "title": str, 특허명
    "abs_cont": str, 개요(초록)
    "claims": str, 청구항
    "technical_field": str, 기술분야
    "background_art": str, 발명의 배경
    "problem": str, 문제상황
    "solution": str, 해결방법
    "effects": str, 발명으로 인한 효과
    "description": str, 세부설명
    "applicability": str, 활용방안
```