from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def evaluate(all_pred, all_label):
    acc = 0
    for i in range(len(all_pred)):
        if(all_pred[i] == all_label[i]):
            acc += 1
        else:
            continue
    acc = acc / len(all_pred)
    return acc