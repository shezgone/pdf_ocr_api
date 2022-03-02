import os
import io
import cv2
import pdfplumber
import uvicorn
import pathlib
import platform
import json
import uuid
import re
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form
from fastapi.responses import FileResponse
from starlette.responses import RedirectResponse, StreamingResponse
from fastapi.logger import logger
from typing import List, Optional
from pydantic import BaseModel

import nms_matching

description = """
PDF OCR API는 ncloud OCR상품을 보완하기 위한 실험적 API입니다. 🚀
"""

app = FastAPI(
    title="PDF OCR API",
    description=description,
    version="1.0.0",
    contact={
        "name": "Cloud Solution Architect",
        "url": "https://ncloud.com",
        "email": "kim.yongmin@navercorp.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


@app.post("/slap/v1/pdf-ocr")
async def detect_pdf_ocr_extract_table(
        value_fields: Optional[List[str]] = Query(
            ['1 신고자', '2 수출대행자', '3 제조사', '4 구매자', '5 신고번호', '6 세관.과', '7 신고일자', '8 신고구분H', '9 C/S구분',
             '10 거래구분', '11 종류', '12 결제방법', '13 목적국', '14 적재항', '15 선박회사',
             '16 선박명(항공편명)', '17 출항예정일자', '18 적재예정보세구역', '19 운송형태', '20 검사희망일',
             '21 물품소재지', '22 L/C번호', '23 물품상태', '24 사전임시개청통보여부', '25 반송사유', '26 환급신청인',
             '27 품명', '28 거래품명', '29 상표명']),
        line_fields: Optional[List[str]] = Query(['●품명ㆍ규격', '30모델ㆍ규격', '(NO.']),
        next_fields: Optional[List[str]] = Query(
            ['35 세번부호', '21 물품소재지', '44 총중량', '45 총포장갯수', '(FOB)', '57 신고수리일자']),
        pdf_file: UploadFile = File(...)):
    """
    파리미터 값은 추가, 수정, 삭제가 가능합니다.
    수출입원장 같은 **구조적인 PDF문서를 분석**하는데 유용합니다.
    - **value_fields** : 필드안에 값이 포함되어 있는 경우 검색어를 입력합니다.
    - **line_fields** : 검색어로 시작하는 경우 레코드를 반환합니다.
    - **next_fields** : 필드 다음 필드에 값이 존재하는 경우 검색어를 입력합니다.
    - **pdf_file** : 검색 대상 pdf 파일입니다.
    """
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, detail="Invalid document type")

    contents = await pdf_file.read()

    with open(os.path.join("./", pdf_file.filename), "wb") as fp:
        fp.write(contents)

    pages = []

    with pdfplumber.open(pdf_file.filename) as pdf:
        for page in pdf.pages:
            tab = page.extract_table(table_settings={"keep_blank_chars": True})

            for line in tab:
                logger.info([i for i in line if i is not None])

            result_dic = {}

            for idx, qry in enumerate(value_fields + line_fields + next_fields):
                i = 1
                for rec in tab:
                    rec = [x for x in rec if x is not None]

                    for jdx, word in enumerate(rec):
                        if qry.replace(" ", "") in word.replace(" ", ""):

                            if idx < len(value_fields):
                                try:
                                    result = word.replace(qry, "").replace(qry.replace(" ", ""), "")
                                    if result.startswith("\n"):
                                        result = result[1:]
                                    result_dic[qry] = result
                                except IndexError as e:
                                    pass
                                continue
                            elif idx < len(value_fields) + len(line_fields):
                                result_dic[qry + ":" + str(i)] = rec
                                i += 1
                                continue
                            else:
                                try:
                                    if rec[jdx + 1] != '':
                                        result_dic[qry] = rec[jdx + 1]
                                    else:
                                        result_dic[qry] = rec[jdx + 2]
                                except Exception as e:
                                    logger.error("Next field error: {0}".format(e))
                                    # raise HTTPException(400, detail="Next field error")
                                continue

            pages.append(result_dic)

    if os.path.isfile(pdf_file.filename):
        os.remove(pdf_file.filename)

    return {"ocr_result": pages}


@app.post("/edutech/v1/pdf-ocr")
async def detect_pdf_ocr_extract_words(start_fields: Optional[List[str]] = Query(['기출', '끝장']),
                                       end_fields: Optional[List[str]] = Query(['정답', '①②③④⑤']),
                                       pdf_file: UploadFile = File(...),
                                       return_type: str = Query("json", enum=["json", "image"])):
    """
        시작 단어 2개와 끝나는 단어 2개 사이의 텍스트 영역안에 텍스트를 추출합니다.
        - **start_fields** : 시작하는 문자열의 조합을 입력합니다. 현재 2개로 설정.
        - **end_fields** : 끝나는 문자열의 조합을 입력합니다. 현재 2개로 설정.
        - **pdf_file** : 검색 대상 pdf 파일입니다.
        - **return_type** : Json이나 이미지로 결과값을 리턴합니다. (기본값: json)
    """
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, detail="Invalid document type")

    contents = await pdf_file.read()
    with open(os.path.join("./", pdf_file.filename), "wb") as fp:
        fp.write(contents)

    with pdfplumber.open(pdf_file.filename) as pdf:

        pages = {}

        for i, page in enumerate(pdf.pages):
            bboxs = []
            x0, top, x1, bottom = [], [], [], []
            results = []
            for rec1, rec2 in zip(page.extract_words()[:-1], page.extract_words()[1:]):

                if rec1["text"].replace(" ", "") in start_fields[0] and \
                        rec2["text"].replace(" ", "") in start_fields[1]:
                    x0.append(rec1["x0"])
                    top.append(rec1["top"])

                if rec1["text"].replace(" ", "") in end_fields[0] and \
                        rec2["text"].replace(" ", "") in end_fields[1]:
                    x1.append(rec2["x1"])
                    bottom.append(rec2["bottom"])

            # x0, top <-> x1, bottom 개수의 차이는 작은쪽으로 맞춘다.
            for bbox in zip(x0, top, x1, bottom):
                bboxs.append(bbox)

            if return_type == 'json':
                for bbox in bboxs:
                    results.append({"bbox": bbox, "text": page.crop(bbox).extract_text()})
            else:
                if len(bboxs) != 0:
                    im = page.to_image()
                    im_box = im.draw_rects(bboxs)
                    file_name = "img_bbox_" + str(i) + ".jpg"
                    im_box.save(file_name)
                    cv2img = cv2.imread(file_name)
                    res, im_png = cv2.imencode(".png", cv2img)
                    if os.path.isfile(file_name):
                        os.remove(file_name)

                    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

        logger.info(results)
        pages["page_" + str(i)] = results

    if os.path.isfile(pdf_file.filename):
        os.remove(pdf_file.filename)

    return {"ocr_results": pages}


@app.post("/edutech/v1/pdf-ocr-regex")
async def detect_pdf_ocr_extract_words_regex(regex: Optional[List[str]] = Query(["기출,끝장", "정답,①|②|③|④|⑤"]),
                                             pdf_file: UploadFile = File(...),
                                             return_type: str = Query("json", enum=["json", "image"])):
    """
        시작과 끝을 정의하지 않고 정규식 박스에서 MAX(x0, top), MIN(x1, bottom)으로 영역을 구합니다..
        - **regex** : 정규표현식 (2Gram: 각 표현식은 콤마로 구분)
        - **pdf_file** : 검색 대상 pdf 파일입니다.
        - **return_type** : Json이나 이미지로 결과값을 리턴합니다. (기본값: json)
    """
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, detail="Invalid document type")

    contents = await pdf_file.read()
    with open(os.path.join("./", pdf_file.filename), "wb") as fp:
        fp.write(contents)

    with pdfplumber.open(pdf_file.filename) as pdf:

        pages = {}

        reg_list = []
        for query in regex:
            q1 = query.split(",")[0]
            q2 = query.split(",")[1]
            p1 = re.compile(q1.replace(" ", ""))
            p2 = re.compile(q2.replace(" ", ""))
            reg_list.append((p1, p2))

        for i, page in enumerate(pdf.pages):
            bboxs = []
            results = []
            group_id = 0
            group_keys = []
            for rec1, rec2 in zip(page.extract_words(use_text_flow=False)[:-1],
                                  page.extract_words(use_text_flow=False)[1:]):

                logger.info(rec1)

                for p_idx, p in enumerate(reg_list):
                    # Bi-Grams Match 만족
                    if p[0].match(rec1["text"].replace(" ", "")) and p[1].match(rec2["text"].replace(" ", "")):
                        bboxs.append([p_idx, int(rec1["x0"]), rec1["x0"], rec1["top"], rec2["x1"], rec2["bottom"]])
                        print(bboxs)

            col_name = ['group', 'x0_int', 'x0', 'top', 'x1', 'bottom']
            bboxs_df = pd.DataFrame(bboxs, columns=col_name).sort_values(by=['group', 'x0_int', 'top'])

            df = bboxs_df.sort_values(by=["x0_int", "top"], ascending=True)
            df['rank'] = tuple(zip(df.x0_int, df.top))
            bboxs_df['row_num'] = df.groupby('group', sort=False)['rank'].apply(
                lambda x: pd.Series(pd.factorize(x)[0])).values

            print(bboxs_df)

            # return {'ok'}

            bboxs_df = bboxs_df.groupby('row_num').agg(
                cnt=pd.NamedAgg(column='x0', aggfunc='count'),
                x0=pd.NamedAgg(column='x0', aggfunc='min'),
                top=pd.NamedAgg(column='top', aggfunc='min'),
                x1=pd.NamedAgg(column='x1', aggfunc='max'),
                bottom=pd.NamedAgg(column='bottom', aggfunc='max')).reset_index()

            temp = bboxs_df.copy()
            mask = temp['cnt'] > 1
            bboxs_df = temp.loc[mask, :]

            bboxs = []

            for rec in bboxs_df.itertuples():
                bboxs.append((rec.x0, rec.top, rec.x1, rec.bottom))

            if return_type == 'json':
                for bbox in bboxs:
                    try:
                        results.append({"bbox": bbox, "text": page.crop(bbox).extract_text()})
                    except ValueError as e:
                        logger.error("Negative Bbox error: {0}".format(e))
            else:
                if len(bboxs) != 0:
                    im = page.to_image()
                    im_box = im.draw_rects(bboxs, stroke='green')
                    file_name = "img_bbox_" + str(i) + ".jpg"
                    im_box.save(file_name)
                    cv2img = cv2.imread(file_name)
                    res, im_png = cv2.imencode(".png", cv2img)
                    if os.path.isfile(file_name):
                        os.remove(file_name)

                    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

        logger.info(results)
        pages["page_" + str(i)] = results

    if os.path.isfile(pdf_file.filename):
        os.remove(pdf_file.filename)

    return {"ocr_results": pages}


@app.post("/edutech/v1/pdf-ocr-nms")
async def detect_pdf_ocr_nms(from_tag: Optional[List[str]] = Query(["91", "86", "159", "116"]),
                             to_tag: Optional[List[str]] = Query(["504", "276", "530", "300"]),
                             pdf_file: UploadFile = File(...),
                             sample_page: str = Query("3"),
                             nms_confidence: str = Query("0.8"),
                             return_type: str = Query("json", enum=["json", "image"])):
    """
        시작 이미지 Bbox와 끝나는 이미지 Bbox 사이의 텍스트 영역안에 텍스트를 추출합니다.
        - **from_tag** : 시작하는 이미지 위치(x1, y1(top), x2, y2(bottom)).
        - **to_tag** : 끝나는 이미지 위치(x1, y1(top), x2, y2(bottom)).
        - **pdf_file** : 검색 대상 pdf 파일입니다.
        - **sample_page** : tag위치가 정의된 샘플 페이지 번호로 설정.
        - **nms_confidence** : template matching 정확도 default: 0.8
        - **return_type** : Json이나 이미지로 결과값을 리턴합니다. (기본값: json)
    """
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, detail="Invalid document type")

    from_tag = list(map(int, from_tag))
    to_tag = list(map(int, to_tag))

    if from_tag[2] < from_tag[0] or from_tag[3] < from_tag[1] or to_tag[2] < to_tag[0] or to_tag[3] < to_tag[1]:
        raise HTTPException(400, detail="[Input Parameters] Negative of Bbox is not allowed")

    sample_page = int(sample_page) - 1
    logger.info("NMS Sample page: {0}".format(sample_page))

    contents = await pdf_file.read()
    with open(os.path.join("./", pdf_file.filename), "wb") as fp:
        fp.write(contents)

    with pdfplumber.open(pdf_file.filename) as pdf:
        pdf.pages[sample_page].crop(from_tag).to_image().save('from_tag.png')
        pdf.pages[sample_page].crop(to_tag).to_image().save('to_tag.png')
        from_tag = cv2.imread('from_tag.png')
        to_tag = cv2.imread('to_tag.png')

        pages = {}

        for i, page in enumerate(pdf.pages):

            page.to_image().save('pages_' + str(i) + ".png")
            page_img = cv2.imread('pages_' + str(i) + ".png")

            pick_from = nms_matching.get_bbox(page_img, from_tag, threshold=float(nms_confidence))
            pick_to = nms_matching.get_bbox(page_img, to_tag, threshold=float(nms_confidence))

            logger.info("NMS pick_from: {0}".format(pick_from))
            logger.info("NMS pick_to: {0}".format(pick_to))

            bboxs = []
            results = []

            for (x1, y1, _, _), (x2, y2, w2, h2) in zip(sorted(pick_from, key=lambda x: x[1]),
                                                        sorted(pick_to, key=lambda x: x[1])):
                if x1 > x2 or y1 > y2:
                    raise HTTPException(400, detail="[NMS Results] Negative of Bbox is not allowed")
                # float형으로 타입 변환해야 ValueError: [TypeError("'numpy.int64' object is not iterable")를 회피할 수 있음.
                bboxs.append((float(x1), float(y1), float(w2), float(h2)))

            if return_type == 'json':
                for bbox in bboxs:
                    results.append({"bbox": bbox, "text": page.crop(bbox).extract_text()})
            else:
                if len(bboxs) != 0:
                    im = page.to_image()
                    im_box = im.draw_rects(bboxs)
                    file_name = "img_bbox_" + str(i) + ".jpg"
                    im_box.save(file_name)
                    cv2img = cv2.imread(file_name)
                    res, im_png = cv2.imencode(".png", cv2img)
                    if os.path.isfile(file_name):
                        os.remove(file_name)

                    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

        logger.info(results)
        pages["page_" + str(i)] = results

    if os.path.isfile(pdf_file.filename):
        os.remove(pdf_file.filename)

    return {"ocr_results": pages}


class key_bbox(BaseModel):
    left: int = 10
    top: int = 10
    width: int = 20
    height: int = 20


class val_bbox(BaseModel):
    left: int = 10
    top: int = 10
    width: int = 20
    height: int = 20


class Item(BaseModel):
    class_name: str = "Class1"
    sample_page: int = 1
    key: key_bbox
    val: val_bbox


class Item2(BaseModel):
    class_name: str = "Class2"
    sample_page: int = 1
    key: key_bbox
    val: val_bbox


@app.get("/comm/v1/settings_sample")
async def detect_pdf_ocr_sample(item1: Item, item2: Item2):
    return {"Sample JSON": [item1, item2]}


@app.post("/comm/v1/pdf-ocr-template")
async def detect_pdf_ocr_template(nms_confidence: str = Query("0.8"),
                                  json_file: UploadFile = File(..., json={"test": 2}),
                                  pdf_file: UploadFile = File(..., description="분석할 페이지")):
    """
            필드 Bbox에 매핑된 값 Bbox 영역 안에 텍스트를 추출합니다.
            - **nms_confidence** : template matching 정확도 default: 0.8
            - **json_file** : 검색할 정보를 세팅합니다.
            - **pdf_file** : 검색 대상 pdf 파일입니다.
    """
    if not json_file.filename.lower().endswith('.json'):
        raise HTTPException(400, detail="Invalid JSON document type")

    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, detail="Invalid PDF document type")

    tempdir = "./" + str(uuid.uuid4())
    os.mkdir(tempdir)

    contents = await pdf_file.read()
    with open(os.path.join(tempdir, pdf_file.filename), "wb") as fp:
        fp.write(contents)

    contents = await json_file.read()
    with open(os.path.join(tempdir, json_file.filename), "wb") as fp:
        fp.write(contents)

    with open(tempdir + "/" + json_file.filename) as jf:
        settings = json.load(jf)
        print(settings)

    templates = []

    with pdfplumber.open(tempdir + "/" + pdf_file.filename) as pdf:

        for idx, setting in enumerate(settings):
            print(setting["class_name"])
            try:
                pdf.pages[setting["sample_page"] - 1].crop(list((setting["key_bbox"]["left"],
                                                                 setting["key_bbox"]["top"],
                                                                 setting["key_bbox"]["left"] + setting["key_bbox"][
                                                                     "width"],
                                                                 setting["key_bbox"]["top"] + setting["key_bbox"][
                                                                     "height"]
                                                                 ))).to_image().save(
                    tempdir + '/template' + str(idx) + ".png")
            except Exception as e:
                logger.error("Bbox crop error: {0}".format(e))
                raise HTTPException(400, detail="Bbox crop error: {0}".format(e))
            templates.append({
                "class_name": setting["class_name"],
                "key_image": cv2.imread(tempdir + '/template' + str(idx) + ".png"),
                "val_bbox": setting["val_bbox"],
                "key_bbox": setting["key_bbox"]
            })

        pages = {}

        for i, page in enumerate(pdf.pages):

            page.to_image().save(tempdir + '/pages_' + str(i) + ".png")
            page_img = cv2.imread(tempdir + '/pages_' + str(i) + ".png")

            results = []
            for item in templates:
                pick = nms_matching.get_bbox(page_img, item['key_image'],
                                             threshold=float(nms_confidence))

                print(pick)

                for x, y, _, _ in sorted(pick, key=lambda x: x[1]):
                    try:
                        results.append({"class_name": item['class_name'],
                                        "text": page.crop((x + item['val_bbox']["left"] - item['key_bbox']["left"],
                                                           y + item['val_bbox']["top"] - item['key_bbox']["top"],
                                                           x + item['val_bbox']["left"] - item['key_bbox']["left"] +
                                                           item['val_bbox']["width"],
                                                           y + item['val_bbox']["top"] - item['key_bbox']["top"] +
                                                           item['val_bbox']["height"],
                                                           )).extract_text()})
                    except Exception as e:
                        logger.error("Results append crop extract text error: {0}".format(e))
                        pass
            pages["page_" + str(i + 1)] = results

    return {"ocr_results": pages}


@app.get("/download_samples")
async def download_pdfs():
    file_path = "samples/samples.zip"
    return FileResponse(path=file_path, filename="samples.zip", media_type='application/octet-stream')


@app.get("/health-check")
def check_alive():
    return {"status": "ok"}


@app.get("/")
async def redirect_home():
    response = RedirectResponse(url='/docs')
    logger.info("direction run!")
    return response


if __name__ == "__main__":
    cwd = pathlib.Path(__file__).parent.resolve()
    if platform.system() == 'Linux':
        uvicorn.run("__main__:app", host="0.0.0.0", port=8000, log_config=f"{cwd}/log.ini", workers=5)
    else:
        uvicorn.run("__main__:app", host="0.0.0.0", port=8000, log_config=f"{cwd}/log.ini", reload=True)
