mbti_types = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP","ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"]

def extract_mbti_and_conversations(input_file_path):
    with open(input_file_path, "r", encoding="utf-8") as file:
        conversation = ""  
        with open("conversation_data.txt", "w", encoding="utf-8") as output_file:
            for line in file:
                # 대화 시작인지 확인, 그 줄에 ] 도 같이 있는지
                if line.startswith("[") and "]" in line:

                   #ex) [Jinny/ENTP] [오전 11:11] 배고파요 { '[' 기준으로 나눔} => ["", "Jinny/ENTP]", "오전 11:11] 배고파요"] => [1]에서 비교 
                    temp = line.split("[")
                    mbti_part = temp[1]

                    mbti = ""
                    for i in mbti_types:
                        # ENTP-a 이런 유형은 일단 제외 (-a 빼고 앞에 4글자만 가져오던가...)
                        if i in mbti_part.upper():
                            mbti = i
                            break

                    conversation_part = line.split("] ")[1].strip()
                    #대화 내용 연결
                    conversation += conversation_part + " "
                    
                    # 대화 끝나는지 (다음 [ 나오기 전까지)
                    if "[" in line:
                        output_file.write("MBTI: " + mbti + "\n")
                        output_file.write("Conversation: " + conversation + "\n")
                        
                        # 대화 변수 초기화
                        conversation = ""


input_file_path = "Data\OriginData.txt"
extract_mbti_and_conversations(input_file_path)