import os
import numpy as np


if __name__ == '__main__':
    pred_nb = 1
    cutout_path = "../datasets/InLoc/cutouts_imageonly/"
    query_path = "../datasets/InLoc/iphone7/"
    html = """<html><table border="1">\n"""

    html += "<tr>"
    html += "<th>query</th>"
    for i in range(1,pred_nb+1):
        html += "<th>pred" + str(i) + "</th>"
    html += "</tr>\n"

    predictions = np.load("predictions.npy")
    cutout_imgnames_all = np.load("../datasets/InLoc/cutout_imgnames_all.npy", allow_pickle=True)
    query_imgnames_all = np.load("../datasets/InLoc/query_imgnames_all.npy", allow_pickle=True)
    
    for row in range(predictions.shape[0]):
        html += "<tr>"
        html += "<td><img src='{}' height='200'></td>".format(os.path.join(query_path, query_imgnames_all[row][0]))
        for pred_id in range(pred_nb):
            html += "<td><img src='{}' height='200'></td>".format(os.path.join(cutout_path, cutout_imgnames_all[predictions[row, pred_id]][0]))
        html += "</tr>\n"

    html += "</table></html>"

    file_ = open('result.html', 'w')
    file_.write(html)
    file_.close()
