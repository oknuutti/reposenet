
import matplotlib.pyplot as plt
import sys
import csv
import os
import numpy as np

def main():
    arch = sys.argv[1]
    versions = sys.argv[2].split(',')
    data = {}

    for v in versions:
        d = np.zeros((2000, 16))
        logfile = os.path.join(os.path.dirname(__file__), '..', 'output', 'stats_'+arch+'-'+v+'.csv')
        with open(logfile, newline='') as fh:
            reader = csv.reader(fh, delimiter='\t', quotechar='"')
            for i, row in enumerate(reader):
                if i > 1:
                    d[i-2, :] = list(map(float, row))
        data[v] = d

    titles = {
        'f1': 'lr = 2e-5',
        'f2': 'lr = 5e-5',
        'f3': 'lr = 1e-4',
        'f4': 'lr = 2e-4',
        'f5': 'lr = 5e-4',
        'af3': 'adversarial training, lr = 1e-4',
    }

    plt.rcParams.update({'font.size': 22})
    fig, axs = plt.subplots(2, 3, figsize=(30, 19.5), sharex=True, sharey=True)
    axs = axs.flatten()
    for i, v in enumerate(versions):
        ax = axs[i]

        # texts
        ax.set_title(titles[v])
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')

        # limits & ticks
        maxx, ticx = 2000, 250
        maxy, ticy = 20, 5
        ax.set_xlim(0, maxx)
        ax.set_ylim(0, maxy)
        ax.set_xticks(range(0, maxx+ticx, ticx))
        ax.set_yticks(range(0, maxy+ticy, ticy))
        ax.vlines(range(0, maxx+ticx, ticx), 0, maxy, '0.7', '--')
        ax.hlines(range(0, maxy+ticy, ticy), 0, maxx, '0.7', '--')

        # plotting
        sc = 1.6 if arch == 'goog' else 1
        label_postfix = '/1.6' if arch == 'goog' else ''
        ax.plot(data[v][:, 0], data[v][:, 1]/sc, ':', label='training'+label_postfix)
        ax.plot(data[v][:, 0], data[v][:, 6], '-', label='validation')
        ax.legend()

    plt.tight_layout()
    while(not plt.waitforbuttonpress()):
        pass



if __name__ == '__main__':
    main()