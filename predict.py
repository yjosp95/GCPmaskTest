from requests import get
# import pandas as pd
# from google.cloud import automl_v1beta1
# from google.cloud.automl_v1beta1.proto import service_pb2
import cv2
import time
import numpy as np

def get_prediction(content, project_id, model_id):
    prediction_client = automl_v1beta1.PredictionServiceClient()


    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    payload = {'image': {'image_bytes': content }}
    params = {}
    request = prediction_client.predict(name, payload, params)

    return request  # waits till request is returned

# 라즈베리파이에서 사진 저장
if __name__ == '__bb__':
    img = cv2.VideoCapture('http://223.194.46.22:8080/?action=stream')
    i=0
    while True:
        ret, frame = img.read()
        cv2.imwrite('img'+str(i)+'.jpg', frame)
        i = i + 1
        print(i)

# 출력결과 동영상처럼 보이기
if __name__ == '__main__':
    i = 1
    ctime = time.time()
    while i < 100 :
        time.sleep(0.2)
        img = cv2.imread('./pred'+str(i)+'.jpg')
        i = i+1

        cv2.imshow('title', img)
        cv2.waitKey(1)
    print(time.time() - ctime)
    cv2.destroyAllWindows()

# 사진에서 결과 뽑기
if __name__ == '__dd__':
    project_id = '854715581202'
    model_id = 'IOD1112250569495412736'

    url = "http://223.194.46.22:8080/?action=snapshot" #"라즈베리파이 ip"/?action=snapshot

    file_name = "snapshot.jpg"      # snapshot을 저장할 파일 이름

    i=0

    while True:
        ctime = time.time()
        # ff = get(url)   # url을 통해 파일 얻기
        # content = ff.content    # 얻어온 파일 내용 얻기
        #
        # with open(file_name, "wb") as file: #파일 저장
        #     file.write(content)
        fn = './img'+str(i)+'.jpg'
        i = i+1
        a = cv2.imread(fn)

        with open(fn, 'rb') as ff:
            content = ff.read()

        result = get_prediction(content, project_id, model_id)
        print(i)
        print(result)


        for res in result.payload :
            start_x = res.image_object_detection.bounding_box.normalized_vertices[0].x * 720
            start_y = res.image_object_detection.bounding_box.normalized_vertices[0].y * 480
            end_x = res.image_object_detection.bounding_box.normalized_vertices[1].x * 720
            end_y = res.image_object_detection.bounding_box.normalized_vertices[1].y * 480
            if "{}".format(res.display_name) == "mask":
                cv2.circle(a, (int((start_x + end_x)/2),int((start_y + end_y)/2)), int((end_x-start_x)/2), (0, 255,0), 3)
            else :
                cv2.line(a, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0,0,255), 3)
                cv2.line(a, (int(start_x), int(end_y)), (int(end_x), int(start_y)), (0,0,255), 3)

        cv2.imwrite('pred'+str(i)+'.jpg', a)

        cv2.imshow('title', a)
        cv2.waitKey(1)
        print(time.time() - ctime)
    cv2.destroyAllWindows()
