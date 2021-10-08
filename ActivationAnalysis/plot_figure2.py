import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

file_name = './Frequency/cifar10_cas_layer4_class0.npy'

res = np.load(file_name)

res = res.reshape(2, 512)
statis_results_robust = res[0]
statis_results_std = res[1]

save_name = 'cifar10_resnet18_cas_layer4_clip_1e-2'

fig = plt.figure(figsize=(8, 7))
predict_res_1 = statis_results_std
predict_res_2 = statis_results_robust
index = np.argsort(-predict_res_1)
# index = np.argsort(-predict_res_2)
predict_res_1 = predict_res_1[index]
predict_res_2 = predict_res_2[index]



left, bottom, width, height = 0.2, 0.15, 0.78, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
ax1.set_ylim(0, 1000)

ax1.bar(range(len(predict_res_1)), predict_res_1, color='royalblue', label='natural examples')
ax1.bar(range(len(predict_res_1)), predict_res_2, color='indianred', alpha=0.7, label='adversarial examples')

ax1.legend(loc=0, ncol=1, prop={'family': 'Times Roman', 'size': 24})
ax1.grid(linestyle='--', color='lightgray')
ax1.set_ylabel('Number of Activation', fontdict={'size': 28})
ax1.set_xlabel('Channel', fontdict={'size': 28})
ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.tick_params(axis='both', which='minor', labelsize=24)


if os.path.exists('./Figure2') == False:
    os.makedirs('./Figure2')
plt.savefig('./Figure2/%s.pdf' % (save_name))
plt.savefig('./Figure2/%s.png' % (save_name))