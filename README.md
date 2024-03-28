### A chatbot service that users can talk to an AI based on 16 MBTI types.

## 프로젝트 개요


## LLAMA 1 / LLAMA 2 논문 분석
- ###  Pretrained data: LLAMA 2 에서는 LLAMA 1과 거의 비슷한 pretraining setting 과 model architectue를 사용하였다.
  
 >  ![image](https://github.com/JinnyKo/Skippy_mbti_Llama2/assets/93627969/6160bdf0-b82d-4c6a-b08c-24d3aab9053b)

     (LLAMA1 pretraining Data)
     - CommonCrawl (67.0%): Language Classification 해서 영어로 된 데이터만 사용 
     - Github (4.5%): 코드 데이터 학습, Apache 라이센스, MIT 라이센스 
     
- ### Result "Common Senese Reasoning" 상식적으로 추론하는 성능

 > ![image](https://github.com/JinnyKo/Skippy_mbti_Llama2/assets/93627969/85a952ef-671e-4c98-b957-1f333ccbd5bd)

  



## References 
- https://arxiv.org/pdf/2302.13971.pdf (LLAMA 1 paper: LLaMA: Open and Efficient Foundation Language Models
Hugo Touvron∗
, Thibaut Lavril∗
, Gautier Izacard∗
, Xavier Martinet
Marie-Anne Lachaux, Timothee Lacroix, Baptiste Rozière, Naman Goyal
Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin
Edouard Grave∗
, Guillaume Lample∗
Meta AI) 
