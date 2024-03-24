import traceback
import os 

def extract_mbti_and_conversations(input_file_path):
    mbti_types = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"]
    with open(input_file_path, "r", encoding="utf-8") as file:
        conversation = ""  
        with open(os.path.join("Data", "output1_data.txt"), "w", encoding="utf-8") as output_file:
            # 줄 번호와 함께 파일을 반복하여 오류가 발생하는 줄을 식별
            for line_number, line in enumerate(file, start=1):
                try:
                    # 대화 시작인지 확인, 그 줄에 ]도 같이 있는지
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

                        conversation_part = line.split("]")[2].strip()  
                        # 대화 내용 연결
                        conversation += conversation_part + " "
                        
                        # 대화가 끝나는지 확인 후 초기화 (다음 [ 나오기 전까지)
                        if "[" in line:
                            output_file.write("MBTI: " + mbti + "\n")
                            output_file.write("Conversation: " + conversation + "\n")
                            conversation = ""
                except IndexError as e:
                    print(f"Error in line {line_number}: {e}")
                    # traceback 모듈을 사용, 전체 오류 출력
                    traceback.print_exc()

input_file_path = "Data\OriginData.txt"
extract_mbti_and_conversations(input_file_path)

def modify_conversation_format(input_file_path, output2_file_path):
    with open(input_file_path, "r", encoding="utf-8") as file:
        with open(output2_file_path, "w", encoding="utf-8") as output_file:
            mbti = None
            for line in file:
                if line.startswith("MBTI:"):
                    mbti = line.split(":")[1].strip()
                elif "이모티콘" not in line:
                    conversation = line.split("Conversation:")[1].strip()
                    output_file.write(f"{mbti} {conversation}\n")

output2_file_path = os.path.join("Data", "output2_data.txt")
output1_file_path = os.path.join("Data","output1_data.txt")
modify_conversation_format(output1_file_path,output2_file_path)

def mbti_count(output1_file_path):
    mbti_counts = {} 
    with open(output1_file_path,"r", encoding="utf-8") as file:
        for line in file: 
            if line.startswith("MBTI:"):
                mbti_type = line.split(":")[1].strip()
                if mbti_type in mbti_counts:
                    mbti_counts[mbti_type] += 1
                else:
                    mbti_counts[mbti_type] = 1
    print(mbti_counts)

output1_file_path = os.path.join("Data","output1_data.txt")
mbti_counts = mbti_count(output1_file_path)
