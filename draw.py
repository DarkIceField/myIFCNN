import matplotlib.pyplot as plt
import numpy as np


def drawPicture(picList):
    print(picList.shape)
    picList = picList.squeeze()
    print(picList.shape)
    for i in range(len(picList)):
        newPic = picList[i].numpy()
        print(i)
        ax = plt.subplot(8,8,1+i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(newPic,cmap='Greys')
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.show()

def drawLineChart(xlist,ylabels = None,note=False,title='',*ylists):
    colorArray = ['red','orange','yellow','green','blue','purple','black']
    for i in range(len(ylists)):
        plt.plot(xlist,ylists[i],c=colorArray[i%len(colorArray)],label= None if ylabels is None else ylabels[i],marker='o')
        if note:
            for x,y in zip(xlist,ylists[i]):
                plt.text(x,y,y,ha='center',va='bottom',fontsize=10)

    plt.legend()
    plt.title(title)
    # plt.savefig(fname='result.png')
    plt.show()