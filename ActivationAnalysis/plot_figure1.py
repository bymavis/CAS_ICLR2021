import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

file_name = './Manitude/cifar10_cas_layer4_class0.npy'

res = np.load(file_name)
res = res.reshape(2, 512)
statis_results_robust = res[0]
statis_results_std = res[1]

save_name = 'value_cifar10_resnet18_cas_train_layer4'
fig = plt.figure(figsize=(8, 7))
left, bottom, width, height = 0.15, 0.15, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
ax1.set_ylim(0, 1.0)

predict_res_1 = np.sort(statis_results_std)[::-1]
predict_res_2 = np.sort(statis_results_robust)[::-1]
print(predict_res_2[:10])
print(predict_res_1[:10])
# assert False

xaxis = range(len(predict_res_2))
ax1.bar(xaxis, predict_res_2, color='orange', label='adversarial examples')
ax1.bar(xaxis, predict_res_1, color='blue', alpha=0.5, label='natural examples')
ax1.set_xlabel('Channel', fontdict={'family': 'Times Roman', 'size': 28})
ax1.set_ylabel('Magnitude of activation', fontdict={'family': 'Times Roman', 'size': 28})
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1], loc=0, ncol=1, prop={'family': 'Times Roman', 'size': 24})
ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.tick_params(axis='both', which='minor', labelsize=24)
# plt.legend()

if os.path.exists('./Figure1') == False:
    os.makedirs('./Figure1')

plt.savefig('./Figure1/%s.png' % (save_name))
plt.savefig('./Figure1/%s.pdf' % (save_name))
