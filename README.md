### A chatbot service that users can talk to an AI based on 16 MBTI types.

## 프로젝트 개요
- ### LLAMA 모델을 활용하여 16개의 MBTI 타입 별 한국어 채팅 데이터를 학습시키고, 파인튜닝을 통해 각 MBTI 성격과 특성의 맞게 응답하는 모델 학습. 
   

## LLAMA 1 / LLAMA 2 논문 분석
- ###  Pretrained data: LLAMA 2 에서는 LLAMA 1과 거의 비슷한 pretraining setting 과 model architectue를 사용하였다.
  
 >  ![image](https://github.com/JinnyKo/Skippy_mbti_Llama2/assets/93627969/6160bdf0-b82d-4c6a-b08c-24d3aab9053b)

     (LLAMA1 pretraining Data)
     - CommonCrawl (67.0%): Language Classification 해서 영어로 된 데이터만 사용 
     - Github (4.5%): 코드 데이터 학습, Apache 라이센스, MIT 라이센스 
     
- ### Result "Common Senese Reasoning" 상식적으로 추론하는 성능 [LLAMA 1 ] 

 > ![image](https://github.com/JinnyKo/Skippy_mbti_Llama2/assets/93627969/85a952ef-671e-4c98-b957-1f333ccbd5bd)

      - 비교적 적은 파라미터로도 파라미터 수가 압도적으로 많은 모델들의 성능에 비해 대적할 만 하다.
      
- ### Overall performances
    > ![image](https://github.com/JinnyKo/Skippy_mbti_Llama2/assets/93627969/1b29b440-b111-4fd9-86fb-ee925f238f7e)
 

  
- 인퍼런스비용에 대하여.. 








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
