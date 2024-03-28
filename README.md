### A chatbot service that users can talk to an AI based on 16 MBTI types.

###  프로젝트 개요
- #### LLAMA 모델을 활용하여 16개의 MBTI 타입 별 한국어 채팅 데이터를 학습시키고, 파인튜닝을 통해 각 MBTI 성격과 특성의 맞게 응답하는 모델 학습. 
   

### LLAMA 1 / LLAMA 2 논문
- ####  Pretrained data: LLAMA 2 에서는 LLAMA 1과 거의 비슷한 pretraining setting 과 model architectue를 사용하였다.
  
 >  ![image](https://github.com/JinnyKo/Skippy_mbti_Llama2/assets/93627969/6160bdf0-b82d-4c6a-b08c-24d3aab9053b)

     (LLAMA1 pretraining Data)
     - CommonCrawl (67.0%): Language Classification 해서 영어로 된 데이터만 사용 
     - Github (4.5%): 코드 데이터 학습, Apache 라이센스, MIT 라이센스 
 > ![image](https://github.com/JinnyKo/Skippy_mbti_Llama2/assets/93627969/3ee947a1-aa10-4cfb-88b2-335c1dbc00e6)

    - LLAMA 1 의 논문에서 truthful result를 보면 올바른 답을 할 확률 이 낮은것을 지적하면서, 
    hallucinations을 극복하진 못했다고 언급하고 있는데, LLAMA 2는 이를 극복하기 위한 new data set 을 구성한것 같다. 
    
> ![image](https://github.com/JinnyKo/Skippy_mbti_Llama2/assets/93627969/7652f7d5-de05-4a71-ac0e-0e0c35c2b298)
    - LLAMA2 에서도 어쨌든,,완벽하게 해결되진 않을 것 같다. 
 
     
- #### Result "Common Senese Reasoning" 상식적으로 추론하는 성능 [LLAMA 1 ] 

 > ![image](https://github.com/JinnyKo/Skippy_mbti_Llama2/assets/93627969/85a952ef-671e-4c98-b957-1f333ccbd5bd)

      - 비교적 적은 파라미터로도 파라미터 수가 압도적으로 많은 모델들의 성능에 비해 대적할 만 하다.
      
- #### Overall performances
    > ![image](https://github.com/JinnyKo/Skippy_mbti_Llama2/assets/93627969/1b29b440-b111-4fd9-86fb-ee925f238f7e)
 

  
- 인퍼런스비용에 대하여.. 


### LLaMA 1 vs GPT 
=> 두 모델 모두 Decoder only Transformer 아키텍처 라는 점에선 마이너한 부분들을 제외하면 큰 차이 없음.

- #### LLaMA 1
   - Unsuperviese Pretraining, 이전 까지의 토큰을 이용해서 다음 토큰을 예측하는 방식
   - public data training 
   - 데이터 규모 비교적 큼
   - Size: 7~65B (모델의 사이즈는 줄이고, 데이터는 늘리는)
   - 모델,가중치 오픈 (1의 경우는 학술적 목적에 한해) 
 
- #### GPT
   - Unsupervised pretraining
   - supervised finetuning (사람이 개입해서 finetuning)
   - reinforcement learning w. human feedback  (사람이 개입해서 평가)
   - 상대적으로 손이 많이 가긴 하나, 성능면에선 더 좋음
   - Size: 175B 
   - GPT2 부터는 가중치 오픈 하지 않음, GPT3 부터는 모델 자체도 공개 안됨. 

### Config
![image](https://github.com/JinnyKo/Skippy_mbti_Llama2/assets/93627969/9350c5c4-b23f-4876-b369-615c90af55c3)
(Llama1 config)

![image](https://github.com/JinnyKo/Skippy_mbti_Llama2/assets/93627969/6bc1b095-3ace-4054-8de3-4a0bbbdd5503)
(Llama2 config)





### References 
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
- https://arxiv.org/pdf/2307.09288.pdf ( LLAMA 2 Paper: Llama 2: Open Foundation and Fine-Tuned Chat Models
Hugo Touvron∗ Louis Martin† Kevin Stone†
Peter Albert Amjad Almahairi Yasmine Babaei Nikolay Bashlykov Soumya Batra
Prajjwal Bhargava Shruti Bhosale Dan Bikel Lukas Blecher Cristian Canton Ferrer Moya Chen
Guillem Cucurull David Esiobu Jude Fernandes Jeremy Fu Wenyin Fu Brian Fuller
Cynthia Gao Vedanuj Goswami Naman Goyal Anthony Hartshorn Saghar Hosseini Rui Hou
Hakan Inan Marcin Kardas Viktor Kerkez Madian Khabsa Isabel Kloumann Artem Korenev
Punit Singh Koura Marie-Anne Lachaux Thibaut Lavril Jenya Lee Diana Liskovich
Yinghai Lu Yuning Mao Xavier Martinet Todor Mihaylov Pushkar Mishra
Igor Molybog Yixin Nie Andrew Poulton Jeremy Reizenstein Rashi Rungta Kalyan Saladi
Alan Schelten Ruan Silva Eric Michael Smith Ranjan Subramanian Xiaoqing Ellen Tan Binh Tang
Ross Taylor Adina Williams Jian Xiang Kuan Puxin Xu Zheng Yan Iliyan Zarov Yuchen Zhang
Angela Fan Melanie Kambadur Sharan Narang Aurelien Rodriguez Robert Stojnic
Sergey Edunov Thomas Scialom∗
GenAI, Meta)
