import json
from collections import defaultdict

import matplotlib.pyplot as plt



json_log = "20230125_140901.log.json" 

def load_json_log(json_log):
    log_dict = dict()
    
    with open(json_log, 'r') as log_file:
        for i, line in enumerate(log_file):
            log = json.loads(line.strip())
            if i == 0:
                continue
            if 'epoch' not in log:
                continue
            epoch = log.pop('epoch')
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            for k, v in log.items():
                log_dict[epoch][k].append(v)
    return log_dict


log_dict = load_json_log(json_log)
epochs = list(log_dict.keys())

ys_train, ys_val, xs = [], [], []
for epoch in epochs:
    modes = log_dict[epoch]['mode']
    train_losses = [log_dict[epoch]['loss'][i] for i, mode in enumerate(modes) if mode == 'train']
    val_losses = [log_dict[epoch]['loss'][i] for i, mode in enumerate(modes) if mode == 'val' and i < len(log_dict[epoch]['loss'])]
    train_loss = sum(train_losses) / len(train_losses)
    val_loss = sum(val_losses) / len(val_losses)
    xs.append(epoch)
    ys_train.append(train_loss)
    ys_val.append(val_loss)

plt.xlabel('epoch')
plt.plot(xs, ys_train, label="train_loss")
plt.plot(xs, ys_val, label="val_loss")
plt.legend()
plt.show()