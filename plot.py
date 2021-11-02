def plot_loss(data_dict, path):#dictionary data를 입력으로 받아서 plot
    f,ax = plt.subplots(1,3) 
    ax[0].plot(data_dict['train_loss'], label='Training loss')
    ax[0].plot(data_dict['valid_loss'], label='Validation loss')
    ax[0].legend(frameon=False)

    ax[0].set_title('loss')

    ax[1].plot(data_dict['train_acc'], label='Training acc')
    ax[1].plot(data_dict['valid_acc'], label='Validation acc')
    ax[1].legend(frameon=False)
    ax[1].set_title('accuracy')
    # index+=1
    ax[2].plot(data_dict['lr_list'], label='lr ')
    ax[2].legend(frameon=False)
    ax[2].set_title('learning rate')
    #loss와 accuracy, learning rate를 보여줍니다! 
    f.tight_layout()#layout_을 자동으로 맞춰준다.
#    plt.savefig(f'./plot/{path}',facecolor = "w")#주로 사용하는 코드 
    plt.savefig(path,facecolor = "w")#generalize code 
