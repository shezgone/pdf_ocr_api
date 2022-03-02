import pdfplumber
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import cv2
import pandas as pd

KEEP_BLANK_CHARS = False
N_SAMPLES = 3 # 최소 개체수

if __name__ == "__main__":
    with pdfplumber.open("samples/sample6.pdf") as pdf:
        p0 = pdf.pages[0]
        im = p0.to_image()
        words_obj = p0.extract_words(keep_blank_chars=KEEP_BLANK_CHARS,
                                     x_tolerance=3
                                     )
        im.draw_rects(words_obj, stroke="grey")

        print(p0.extract_text())

        print(words_obj)

        # print(p0.objects)
        word_pos = []
        for item in words_obj:
            word_pos.append([
                item['x0'], item['x1'],
                item['top'], item['bottom']
            ])
        X = np.array(word_pos)

        print(X)

        # Standardization 평균 0 / 분산 1
        scaler = StandardScaler()
        # scaler = MinMaxScaler()

        X = scaler.fit_transform(X)

        print(X)

        clustering = OPTICS(min_samples=N_SAMPLES,
                            cluster_method='dbscan',
                            max_eps=.35,  # StnadardScaler
                            # max_eps=.08, # MinMaxScaler
                            metric="euclidean"
                            ).fit(X)
        print(clustering.labels_)

        result = []
        for cls, item in zip(list(clustering.labels_), words_obj):
            result.append([cls, item['text'], item['x0'], item['x1'], item['top'], item['bottom']])

        print(result)

        col_name = ['group', 'text', 'x0', 'x1', 'top', 'bottom']
        word_df = pd.DataFrame(result, columns=col_name)

        word_df = word_df.groupby('group').agg(
            x0=pd.NamedAgg(column='x0', aggfunc='min'),
            top=pd.NamedAgg(column='top', aggfunc='min'),
            x1=pd.NamedAgg(column='x1', aggfunc='max'),
            bottom=pd.NamedAgg(column='bottom', aggfunc='max')).reset_index()

        print(word_df)

        bboxs = []
        for rec in word_df.itertuples():
            # bboxs.append((rec.x0, rec.top, rec.x1, rec.bottom))
            if rec.group == -1:  # Anomaly
                color = 'red'
                stroke_width = 1
            else:
                color = 'green'
                stroke_width = 2

            im.draw_rects([(rec.x0, rec.top, rec.x1, rec.bottom)], stroke_width=stroke_width, stroke=color)

        im.save("samples/sample_out.png")
        cim = cv2.imread("samples/sample_out.png")
        cv2.imshow("output", cim)
        cv2.waitKey(0)
